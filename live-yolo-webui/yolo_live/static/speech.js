"use strict";
// One utterance at a time. No backlog of directions or status messages.
(() => {
  let enabled = false, current = null, last = null, serial = 0, pendingMessage = null;
  const synth = window.speechSynthesis;
  const notify = (type, detail = {}) => window.dispatchEvent(new CustomEvent(`nav:${type}`, {detail}));
  function finish(item, reason = "finished") {
    if (current !== item) return;
    current = null;
    clearTimeout(item.watchdog);
    notify("speech-end", {kind: item.kind, reason});
  }
  function cancel(kind = null) {
    if (!kind) pendingMessage = null;
    if (!current || (kind && current.kind !== kind)) return;
    const item = current;
    current = null;
    clearTimeout(item.watchdog);
    synth?.cancel();
    notify("speech-end", {kind: item.kind, reason: "cancelled"});
  }
  function say(text, options = {}) {
    const {priority = 2, kind = "message", valid = () => true} = options;
    if (!enabled || document.hidden || !text || !valid()) return false;
    if (priority < 3 && window.voiceControls?.blocksSpeech?.()) return false;
    if (priority < 2 && pendingMessage) return false;
    if (current && current.priority > priority) {
      if (priority >= 2) {
        // Retain at most one short-lived acknowledgement, never directions.
        pendingMessage = {text, options, expires: performance.now() + 8000};
        return true;
      }
      return false;
    }
    if (current && current.text === text && current.kind === kind) return false;
    cancel();
    const item = {id: ++serial, text: String(text), priority, kind, valid};
    current = item;
    notify("speech-start", {priority, kind});
    // The event lets the microphone release before output is spoken.
    const utterance = new SpeechSynthesisUtterance(item.text);
    const voices = synth.getVoices().filter(voice => /^en(?:-|$)/i.test(voice.lang));
    utterance.voice = voices.find(voice => voice.localService) || voices[0] || null;
    utterance.lang = utterance.voice?.lang || "en-US";
    utterance.rate = kind === "guidance" ? 1.1 : 1;
    item.utterance = utterance; // Keep a strong reference until end/cancel.
    utterance.onend = () => finish(item);
    utterance.onerror = event => {
      if (current !== item) return;
      finish(item, "error");
      if (!["interrupted", "canceled"].includes(event.error)) {
        enabled = false;
        notify("speech-error", {message: "Spoken output failed. Use the screen reader or enable voice again."});
      }
    };
    last = {text: item.text, kind, valid};
    item.watchdog = setTimeout(() => {
      if (current !== item) return;
      cancel();
      enabled = false;
      notify("speech-error", {message: "Spoken output stalled. Enable voice again or use your screen reader."});
    }, Math.min(120000, Math.max(15000, item.text.length * 130)));
    try { synth.speak(utterance); }
    catch (error) {
      finish(item, "error");
      enabled = false;
      notify("speech-error", {message: error.message});
      return false;
    }
    return true;
  }
  // A direction is cancelled if its observation expires, changes, or is lost.
  setInterval(() => {
    if (current && !current.valid()) cancel();
    if (pendingMessage && performance.now() > pendingMessage.expires) pendingMessage = null;
    if (!current && pendingMessage && !window.voiceControls?.blocksSpeech?.()) {
      const queued = pendingMessage; pendingMessage = null;
      say(queued.text, queued.options);
    }
  }, 100);
  document.addEventListener("visibilitychange", () => { if (document.hidden) cancel(); });
  window.navSpeech = {
    enable() {
      if (!synth || !window.SpeechSynthesisUtterance) throw new Error("This browser has no speech output. Use its screen reader.");
      enabled = true;
      synth.resume();
    },
    disable() { enabled = false; cancel(); },
    say, cancel,
    repeat() { return last && last.valid() ? say(last.text, {kind: last.kind, valid: last.valid}) : false; },
    get enabled() { return enabled; },
    get speaking() { return current !== null; }
  };
})();