"use strict";

(() => {
  const bridge = window.voiceBridge;
  const sidebar = document.querySelector("aside.settings");
  if (!bridge || !sidebar) return;

  const panel = document.createElement("section");
  panel.innerHTML = `
    <h2>Voice target selection</h2>
    <label for="voice-language">Transcription language</label>
    <select id="voice-language">
      <option value="en">English</option>
      <option value="auto">Automatic language detection</option>
    </select>
    <div class="buttons">
      <button id="voice-record" type="button">Speak target</button>
    </div>
    <p id="voice-status" class="hint" role="status" aria-live="polite">
      Say “Find the chair” or “Find the chair on the left”.
    </p>
    <p id="voice-transcript" class="hint"></p>
    <p class="hint">
      Command matching currently supports English.
      Recording stops after ten seconds.
    </p>
  `;
  sidebar.append(panel);

  const button = panel.querySelector("#voice-record");
  const language = panel.querySelector("#voice-language");
  const status = panel.querySelector("#voice-status");
  const transcript = panel.querySelector("#voice-transcript");

  let sequence = 0;
  let recorder = null;
  let microphone = null;
  let request = null;
  let timer = null;
  let starting = false;
  let processing = false;

  function releaseMicrophone() {
    microphone?.getTracks().forEach(track => track.stop());
    microphone = null;
  }

  function idle() {
    starting = false;
    processing = false;
    button.disabled = false;
    button.textContent = "Speak target";
    language.disabled = false;
  }

  function cancel(message = "Voice request cancelled.") {
    ++sequence;
    clearTimeout(timer);
    request?.abort();
    request = null;

    if (recorder && recorder.state !== "inactive") recorder.stop();
    recorder = null;
    releaseMicrophone();
    idle();
    status.textContent = message;
  }

  async function startRecording() {
    const snapshot = bridge.state();
    if (!snapshot.ready || !snapshot.voiceEnabled) {
      status.textContent = "Start the camera with voice enabled first.";
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      status.textContent = "This browser cannot record microphone audio.";
      return;
    }

    const token = ++sequence;
    const selectedLanguage = language.value;
    starting = true;
    button.disabled = true;
    language.disabled = true;
    transcript.textContent = "";
    status.textContent = "Opening microphone…";

    try {
      const acquired = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: {ideal: 1},
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      if (token !== sequence) {
        acquired.getTracks().forEach(track => track.stop());
        return;
      }
      microphone = acquired;

      const mimeType = [
        "audio/webm;codecs=opus",
        "audio/ogg;codecs=opus",
        "audio/mp4"
      ].find(type => MediaRecorder.isTypeSupported(type));

      if (!mimeType) throw new Error("No supported audio recording format.");

      const current = new MediaRecorder(acquired, {
        mimeType,
        audioBitsPerSecond: 64000
      });
      recorder = current;
      const chunks = [];
      let bytes = 0;

      current.ondataavailable = event => {
        if (token !== sequence || !event.data.size) return;
        bytes += event.data.size;
        if (bytes > 2000000) {
          cancel("Recording too large. Try a shorter command.");
          return;
        }
        chunks.push(event.data);
      };

      current.onerror = () => {
        if (token === sequence) cancel("Microphone recording failed.");
      };

      current.onstop = async () => {
        if (token !== sequence) return;
        clearTimeout(timer);
        releaseMicrophone();
        recorder = null;
        processing = true;
        button.disabled = true;
        button.textContent = "Transcribing…";
        status.textContent = "Understanding your target…";

        try {
          const audio = new Blob(chunks, {type: mimeType});
          if (!audio.size) throw new Error("The recording was empty.");

          const requestId = crypto.randomUUID();
          const abort = new AbortController();
          request = abort;
          timer = setTimeout(() => abort.abort(), 60000);

          const response = await fetch(
            `/api/voice/interpret?language=${encodeURIComponent(selectedLanguage)}`,
            {
              method: "POST",
              headers: {
                "Content-Type": mimeType,
                "X-Voice-Request": "1",
                "X-Request-ID": requestId
              },
              body: audio,
              signal: abort.signal
            }
          );

          const result = await response.json();
          if (token !== sequence) return;
          if (result.request_id !== requestId) {
            throw new Error("Unexpected speech response.");
          }

          transcript.textContent = result.transcript
            ? `Heard: ${result.transcript}` : "";

          const now = bridge.state();
          if (!now.ready || now.token !== snapshot.token) {
            status.textContent =
              "Camera or selection changed. Please repeat the command.";
            return;
          }

          if (!response.ok || result.status !== "resolved") {
            status.textContent = result.message || "Please repeat the command.";
            return;
          }

          bridge.select(result.target, result.horizontal);
          status.textContent = `Searching for ${result.target}${
            result.horizontal ? ` on the ${result.horizontal}` : ""
          }.`;
        } catch (error) {
          if (token === sequence) {
            status.textContent = error.name === "AbortError"
              ? "Speech processing timed out. Please try again."
              : error.message;
          }
        } finally {
          if (token === sequence) {
            clearTimeout(timer);
            request = null;
            idle();
          }
        }
      };

      current.start(250);
      starting = false;
      button.disabled = false;
      button.textContent = "Finish recording";
      status.textContent = "Listening… Say one target object.";

      timer = setTimeout(() => {
        if (token === sequence && current.state === "recording") current.stop();
      }, 10000);
    } catch (error) {
      if (token === sequence) {
        cancel(error.name === "NotAllowedError"
          ? "Microphone permission was denied."
          : error.message);
      }
    }
  }

  button.addEventListener("click", () => {
    if (starting || processing) return;
    if (recorder?.state === "recording") recorder.stop();
    else startRecording();
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && (starting || processing || recorder)) {
      cancel("Voice request cancelled because the page was hidden.");
    }
  });

  window.voiceControls = {cancel};
})();