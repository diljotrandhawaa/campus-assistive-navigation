"use strict";
const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("result"), context = canvas.getContext("2d");
const MAX_AGE_MS = 1500;
const OCR_MAX_AGE_MS = 5000;
let lastOcr = null, ocrAvailable = true;
let generation = 0, running = false, socket = null, stream = null, pending = null;
let revision = 0, configuredRevision = -1, frameId = 0, nextTimer = null, frameCallback = null;
let lastCameraFrame = 0, lastVideoTime = -1, lastCapture = 0, lastResult = null;
let sampleCount = 0, latencyTotal = 0, arrivals = [], hasSnapshot = false;

let voiceEnabled = false;
let voiceSide = null;

let lastSpokenState = null;

function speakDirection(direction) {
  if (!("speechSynthesis" in window)) return;

  // Speak only when the direction changes, not on every camera frame.
  if (!["left", "right", "centered"].includes(direction.state)) return;
  if (direction.state === lastSpokenState) return;

  lastSpokenState = direction.state;

  const messages = {
    left: "Pan camera left",
    right: "Pan camera right",
    centered: "Target reached"
  };

  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(
    new SpeechSynthesisUtterance(messages[direction.state])
  );
}

function status(text, active = false) {
  $("status").textContent = text;
  $("status").classList.toggle("active", active);
}
function instruction(state, text, detail = "") {
  $("direction-card").dataset.state = state;
  $("instruction").textContent = text;
  $("direction-icon").textContent = ({left: "←", right: "→", centered: "◎", lost: "⌕", stale: "◷"})[state] || "◎";
  $("target-detail").textContent = detail || `Following: ${$("target").value}`;
  if (!["left", "right", "centered"].includes(state)) {
    $("marker").hidden = true;
    $("offset").textContent = "—";
  }
}
function clearAnalysis(text = "Waiting for a fresh analysis") {
  lastResult = null;
  $("age").textContent = "—";
  instruction("stale", text);
  canvas.classList.add("stale");
  $("expired").hidden = !hasSnapshot;
  $("expired").textContent = text;
  $("save").disabled = true;
  $("detections").replaceChildren();
  $("object-count").textContent = "—";
  $("frame-state").textContent = text;
}
function stop(message = "Camera stopped") {
  window.voiceControls?.cancel("Voice stopped.");
  voiceEnabled = false;
  ++generation;
  running = false;
  clearTimeout(nextTimer);
  if (frameCallback !== null && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frameCallback);
  frameCallback = null;
  const oldSocket = socket;
  socket = null;
  if (oldSocket) oldSocket.close();
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  $("preview-wrap").hidden = true;
  pending = null;
  configuredRevision = -1;
  $("start").disabled = false;
  $("stop").disabled = true;
  clearAnalysis(message);
  clearOcr(message);
  instruction("idle", message);
  status(message);
  $("fps").textContent = "—";
}

async function refreshCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const selected = stream?.getVideoTracks()[0]?.getSettings().deviceId || $("camera").value;
  const options = [new Option("Default camera", "")];
  devices.filter(d => d.kind === "videoinput").forEach((d, i) => {
    options.push(new Option(d.label || `Camera ${i + 1}`, d.deviceId));
  });
  $("camera").replaceChildren(...options);
  if (options.some(o => o.value === selected)) $("camera").value = selected;
}

function watchVideoFrames(token) {
  if (!video.requestVideoFrameCallback) return;
  frameCallback = video.requestVideoFrameCallback(() => {
    if (!running || token !== generation) return;
    lastCameraFrame = performance.now();
    watchVideoFrames(token);
  });
}

function configure(event) {
  const newTarget = !event || event.target?.id === "target";

  if (event?.target?.id === "target") {
    voiceSide = null;
    window.voiceControls?.cancel("Target changed manually.");
  }

  revision++;

  clearAnalysis(
    newTarget ? "Finding target…" : "Updating settings…"
  );

  clearOcr(
    $("ocr-enabled").checked ? "Waiting for OCR scan" : "OCR off"
  );

  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      type: "configure",
      target: $("target").value,
      horizontal: voiceSide,
      new_target: newTarget,
      confidence: Number($("confidence").value),
      ocr: $("ocr-enabled").checked,
      revision
    }));
  }
}

async function start() {
  stop();
  const token = generation;
  running = true;
  $("start").disabled = true;
  $("stop").disabled = false;
  $("error").textContent = "";
  status("Opening camera…");
  instruction("idle", "Opening camera…");
  sampleCount = 0; latencyTotal = 0; arrivals = []; frameId = 0;
  lastCameraFrame = 0; lastVideoTime = -1; lastCapture = 0;
  for (const key of ["latency", "average", "detector", "fps", "age"]) $(key).textContent = "—";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires HTTPS or http://localhost. Open the forwarded local address.");
    const choice = $("camera").value;
    const cameraStream = await navigator.mediaDevices.getUserMedia({audio: false, video: {
      ...(choice ? {deviceId: {exact: choice}} : {}), width: {ideal: 1280},
      height: {ideal: 720}, frameRate: {ideal: 30, max: 30}
    }});
    if (token !== generation) { cameraStream.getTracks().forEach(t => t.stop()); return; }
    stream = cameraStream;
    video.srcObject = stream;
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (token === generation) stop("Camera disconnected");
    });
    await video.play();
    if (token !== generation) return;
    $("preview-wrap").hidden = false;
    watchVideoFrames(token);
    refreshCameras().catch(() => {});
    status("Connecting to YOLO…");
    const connection = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`);
    socket = connection;
    connection.addEventListener("message", event => {
      if (token !== generation) return;
      try { handleMessage(JSON.parse(event.data), token); }
      catch (error) { $("error").textContent = error.message; stop("Invalid server response"); }
    });
    connection.addEventListener("close", () => {
      if (token === generation) {
        $("error").textContent = "Connection closed. Check the YOLO terminal and port forwarding, then start again.";
        stop("Disconnected");
      }
    });
    connection.addEventListener("error", () => {
      if (token === generation) $("error").textContent = "Could not reach the YOLO server on this page's port.";
    });
  } catch (error) {
    if (token !== generation) return;
    const explanations = {
      NotAllowedError: "Camera access was denied. Allow camera access for this page in the browser.",
      NotFoundError: "No camera found. Connect the USB camera to the computer running this browser.",
      NotReadableError: "The camera could not be opened. Close other apps using it and try again.",
      OverconstrainedError: "That camera is unavailable. Select Default camera and try again."
    };
    $("error").textContent = explanations[error.name] || error.message;
    stop("Camera unavailable");
  }
}

function schedule(delay = 0) {
  clearTimeout(nextTimer);
  if (running) nextTimer = setTimeout(capture, Math.max(0, delay));
}

async function capture() {
  if (!running || pending || socket?.readyState !== WebSocket.OPEN || configuredRevision !== revision) return;
  if (document.hidden || video.readyState < 2 || !video.videoWidth || performance.now() - lastCameraFrame > 800) {
    schedule(150); return;
  }
  const token = generation;
  const image = document.createElement("canvas");
  const scale = Math.min(1, Number($("size").value) / Math.max(video.videoWidth, video.videoHeight));
  image.width = Math.round(video.videoWidth * scale);
  image.height = Math.round(video.videoHeight * scale);
  const frame = {id: ++frameId, revision, captured: performance.now(), image};
  pending = frame;  // Includes JPEG encoding time: never allow a second in-flight frame.
  lastCapture = frame.captured;
  image.getContext("2d").drawImage(video, 0, 0, image.width, image.height);
  const jpeg = await new Promise(resolve => image.toBlob(resolve, "image/jpeg", $("ocr-enabled").checked ? 0.9 : 0.8));
  if (token !== generation || !running || pending !== frame) return;
  if (frame.revision !== revision) { pending = null; schedule(); return; }
  if (!jpeg) { $("error").textContent = "Could not encode a camera frame."; stop("Camera error"); return; }
  if (socket?.readyState !== WebSocket.OPEN) { stop("Disconnected"); return; }
  socket.send(JSON.stringify({type: "frame", id: frame.id, revision: frame.revision}));
  socket.send(jpeg);
}

function handleMessage(data) {
  if (data.type === "ready") {
    $("model").textContent = data.model || "YOLOE";
    const selected = $("target").value;
    if (Array.isArray(data.classes) && data.classes.length) {
      $("target").replaceChildren(...data.classes.map(label => new Option(label, label)));
      if (data.classes.includes(selected)) $("target").value = selected;
    }
    ocrAvailable = data.ocr_enabled === true;
    $("ocr-enabled").disabled = !ocrAvailable;
    if (!ocrAvailable) $("ocr-enabled").checked = false;
    $("ocr-mode").textContent = ocrAvailable
      ? `PP-OCRv5 · at least ${data.ocr_interval || 1}s between scans`
      : "OCR disabled at server startup.";
    $("device").textContent = `Device ${data.device} · inference size ${data.imgsz}`;
        voiceEnabled = data.voice_enabled === true;

    const previousTarget = $("target").value;
    const classes = data.classes;
    if (!Array.isArray(classes) || !classes.length) {
      throw new Error("Server returned no target classes.");
    }

    $("target").replaceChildren(
      ...classes.map(label => new Option(label, label))
    );
    $("target").value = classes.includes(previousTarget)
      ? previousTarget : classes[0];

    configure();
    return;
  }
  if (data.type === "configured") {
    if (data.revision === revision) { configuredRevision = revision; schedule(); }
    return;
  }
  if (data.type === "error") {
    $("error").textContent = data.message || "Inference error";
    stop("Analysis stopped");
    return;
  }
  if (!["result", "busy", "discarded"].includes(data.type) || !pending || pending.id !== data.id) return;
  const frame = pending;
  pending = null;
  if (data.type === "busy") {
    status("GPU busy · retrying");
    schedule(80 + Math.random() * 160);
    return;
  }
  const now = performance.now();
  if (data.type === "result" && data.revision === revision && frame.revision === revision && !document.hidden) {
    // OCR is a labelled snapshot with its own shorter-lived history, never current guidance.
    if (now - lastCameraFrame <= 800) renderOcr(data.ocr, frame);
    if (now - frame.captured <= MAX_AGE_MS && now - lastCameraFrame <= 800) {
      render(data, frame);
      const ms = performance.now() - frame.captured;
      sampleCount++; latencyTotal += ms;
      arrivals.push(performance.now());
      arrivals = arrivals.filter(t => performance.now() - t < 5000);
      $("latency").textContent = `${Math.round(ms)} ms`;
      $("average").textContent = `${Math.round(latencyTotal / sampleCount)} ms`;
      $("detector").textContent = `${data.detector_ms} ms`;
      lastResult = {captured: frame.captured};
      status("Streaming", true);
    } else {
      $("detector").textContent = `${data.detector_ms} ms`;
      clearAnalysis("Result too old — waiting");
      status("Waiting for fresh guidance");
    }
  }
  schedule(Math.max(0, 1000 / Number($("rate").value) - (performance.now() - lastCapture)));
}

function boxLabel(text, left, top, color, fontSize) {
  context.font = `600 ${fontSize}px system-ui`;
  const width = Math.min(canvas.width, context.measureText(text).width + 10);
  left = Math.max(0, Math.min(left, canvas.width - width));
  top = Math.max(0, Math.min(top, canvas.height - fontSize - 9));
  context.fillStyle = color;
  context.fillRect(left, top, width, fontSize + 9);
  context.fillStyle = "#07120b";
  context.fillText(text, left + 5, top + fontSize + 1, width - 10);
}

function render(data, frame) {
  if (data.width !== frame.image.width || data.height !== frame.image.height) throw new Error("Frame dimensions do not match the result.");
  canvas.width = frame.image.width; canvas.height = frame.image.height;
  canvas.hidden = false;
  $("placeholder").hidden = true;
  canvas.classList.remove("stale"); $("expired").hidden = true;
  context.drawImage(frame.image, 0, 0);
  const w = canvas.width, h = canvas.height, direction = data.direction;
  speakDirection(direction);
  const band = direction.state === "centered" ? direction.exit_band : direction.enter_band;
  const fontSize = Math.max(12, Math.round(w / 70));
  context.fillStyle = "rgba(244,216,117,0.12)";
  context.fillRect(w * (0.5 - band), 0, 2 * band * w, h);
  context.strokeStyle = "#f4d875"; context.lineWidth = 2;
  context.beginPath(); context.moveTo(w / 2, 0); context.lineTo(w / 2, h); context.stroke();
  data.detections.forEach((d, i) => {
    const selected = i === direction.target_index;
    const color = selected ? "#6edcdb" : "#b4eb6d";
    const [x1, y1, x2, y2] = d.box;
    context.strokeStyle = color; context.lineWidth = selected ? 3 : 2;
    context.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
    boxLabel(`${d.label} ${d.confidence.toFixed(2)} | ${d.position}`, x1 * w, y1 * h - fontSize - 9, color, fontSize);
    if (selected) {
      const cx = (x1 + x2) / 2 * w, cy = (y1 + y2) / 2 * h;
      context.beginPath(); context.moveTo(cx - 8, cy); context.lineTo(cx + 8, cy);
      context.moveTo(cx, cy - 8); context.lineTo(cx, cy + 8); context.stroke();
    }
  });
  boxLabel("CENTER", w / 2 - 30, 8, "#f4d875", fontSize);
  const bannerSize = Math.max(15, Math.round(w / 44));
  context.fillStyle = "rgba(9,15,8,.85)";
  context.fillRect(0, h - bannerSize - 21, w, bannerSize + 21);
  context.fillStyle = direction.state === "centered" ? "#b4eb6d" : "#fff";
  context.font = `600 ${bannerSize}px system-ui`;
  context.fillText(direction.text, 14, h - 13);
  // instruction(direction.state, direction.text,
  //   `Following: ${$("target").value}${direction.matches > 1 ? ` · ${direction.matches} matches; following one` : ""}`);
  instruction(
    direction.state,
    direction.text,
    `${direction.target_index !== null ? "Following" : "Requested"}: ${
      $("target").value
    }${voiceSide ? ` · ${voiceSide}` : ""} · ${direction.matches} matches`
  );
  if (direction.raw_offset !== null) {
    const offset = direction.raw_offset;
    $("marker").hidden = false;
    $("marker").style.left = `${Math.max(0, Math.min(100, (offset + 0.5) * 100))}%`;
    $("offset").textContent = `${Math.abs(offset * 100).toFixed(1)}% ${offset < 0 ? "left" : "right"}`;
  }
  const rows = data.detections.map((d, i) => {
    const row = document.createElement("div");
    row.className = `detection${i === direction.target_index ? " selected" : ""}`;
    const label = document.createElement("span"), score = document.createElement("strong"), pos = document.createElement("small");
    label.textContent = d.label + (i === direction.target_index ? " · target" : "");
    score.textContent = d.confidence.toFixed(2); pos.textContent = d.position;
    row.append(label, score, pos); return row;
  });
  if (!rows.length) {
    const message = document.createElement("p"); message.className = "hint";
    message.textContent = "No configured objects detected in this frame."; rows.push(message);
  }
  $("detections").replaceChildren(...rows);
  $("object-count").textContent = String(data.detections.length);
  $("frame-state").textContent = `Frame ${frame.id}`;
  hasSnapshot = true;
  $("save").disabled = false;
}

setInterval(() => {
  if (!running) return;
  const now = performance.now();
  if (!video.requestVideoFrameCallback && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime; lastCameraFrame = now;
  }
  if (lastResult) {
    const age = now - lastResult.captured;
    $("age").textContent = `${Math.round(age)} ms`;
    if (age > MAX_AGE_MS) clearAnalysis("Analysis is stale");
  }
  if (lastOcr) {
    const age = now - lastOcr.captured;
    $("ocr-age").textContent = `Age: ${(age / 1000).toFixed(1)} s`;
    if (age > OCR_MAX_AGE_MS) clearOcr("OCR snapshot expired — waiting for a new scan");
  }
  if (lastCameraFrame && now - lastCameraFrame > 800) {
    clearAnalysis("Camera paused — waiting"); clearOcr("Camera paused");
  }
  if (pending && now - pending.captured > 10000) {
    $("error").textContent = "No result for 10 seconds. Check the GB10 terminal, then restart the camera.";
    stop("Response timed out");
  }
  arrivals = arrivals.filter(t => now - t < 5000);
  $("fps").textContent = arrivals.length > 1 ? `${((arrivals.length - 1) / ((now - arrivals[0]) / 1000)).toFixed(1)} / s` : "—";
}, 200);

$("start").addEventListener("click", start);
$("stop").addEventListener("click", () => stop());
$("camera").addEventListener("change", () => { if (running) start(); });
$("target").addEventListener("change", configure);
$("ocr-enabled").addEventListener("change", configure);
$("size").addEventListener("change", configure);
$("confidence").addEventListener("input", () => { $("confidence-value").textContent = Number($("confidence").value).toFixed(2); });
$("confidence").addEventListener("change", configure);
$("save").addEventListener("click", () => {
  if (!lastResult || performance.now() - lastResult.captured > MAX_AGE_MS) return;
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `yolo-frame-${Date.now()}.jpg`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/jpeg", 0.95);
});
document.addEventListener("visibilitychange", () => {
  if (!running) return;
  if (document.hidden) { clearAnalysis("Tab paused"); clearOcr("Tab paused"); } else schedule();
});
window.addEventListener("beforeunload", () => stop());
navigator.mediaDevices?.addEventListener?.("devicechange", () => refreshCameras().catch(() => {}));
refreshCameras().catch(() => {});


function clearOcr(message = "Waiting for OCR scan") {
  lastOcr = null;
  $("ocr-frame").hidden = true;
  $("ocr-empty").hidden = false;
  $("ocr-empty").textContent = message;
  $("ocr-state").textContent = message;
  $("ocr-readings").replaceChildren();
  $("ocr-age").textContent = "Age: —";
  $("ocr-time").textContent = "OCR: —";
}

function renderOcr(ocr, frame) {
  if (!ocr || ocr.status === "skipped") return;
  if (!ocrAvailable || !$("ocr-enabled").checked || ocr.status === "disabled") {
    clearOcr("OCR off"); return;
  }
  if (ocr.status === "error") { clearOcr(ocr.message || "OCR failed"); return; }
  if (performance.now() - frame.captured > OCR_MAX_AGE_MS) {
    clearOcr("OCR result too old — hold the camera steady"); return;
  }
  if (ocr.status !== "ok") return;
  const target = $("ocr-frame"), ctx = target.getContext("2d");
  target.width = frame.image.width; target.height = frame.image.height;
  ctx.drawImage(frame.image, 0, 0);
  const w = target.width, h = target.height, fontSize = Math.max(14, Math.round(w / 65));
  ctx.lineWidth = Math.max(2, Math.round(w / 450));
  const rows = ocr.items.map(item => {
    ctx.strokeStyle = "#6edcdb";
    const points = item.polygon;
    ctx.beginPath();
    points.forEach(([x, y], i) => i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h));
    ctx.closePath(); ctx.stroke();
    ctx.font = `600 ${fontSize}px system-ui`;
    const label = item.text;
    const labelWidth = Math.min(w, ctx.measureText(label).width + 10);
    const x = Math.max(0, Math.min(item.box[0] * w, w - labelWidth));
    const y = Math.max(fontSize + 8, item.box[1] * h);
    ctx.fillStyle = "#6edcdb"; ctx.fillRect(x, y - fontSize - 8, labelWidth, fontSize + 8);
    ctx.fillStyle = "#07120b"; ctx.fillText(label, x + 5, y - 5, labelWidth - 10);
    const row = document.createElement("div"); row.className = "ocr-reading";
    const text = document.createElement("span"), score = document.createElement("strong"), note = document.createElement("small");
    // OCR text is untrusted input. Never insert it as HTML.
    text.textContent = item.text;
    score.textContent = item.confidence.toFixed(2);
    note.textContent = item.position + (item.nearby_object ? ` · near ${item.nearby_object.label} in image` : "");
    row.append(text, score, note); return row;
  });
  if (!rows.length) {
    const empty = document.createElement("p"); empty.className = "hint";
    empty.textContent = "No text passed the recognition threshold. Move closer or hold the camera steady.";
    rows.push(empty);
  }
  $("ocr-readings").replaceChildren(...rows);
  $("ocr-empty").hidden = true; target.hidden = false;
  $("ocr-state").textContent = `Snapshot · frame ${frame.id}${ocr.truncated ? " · some regions skipped" : ""}`;
  $("ocr-time").textContent = `OCR: ${ocr.ms} ms`;
  $("ocr-age").textContent = `Age: ${((performance.now() - frame.captured) / 1000).toFixed(1)} s`;
  lastOcr = {captured: frame.captured};
}




window.voiceBridge = {
  state() {
    return {
      token: `${generation}:${revision}`,
      voiceEnabled,
      ready: (
        running &&
        socket?.readyState === WebSocket.OPEN &&
        configuredRevision === revision
      )
    };
  },

  select(target, horizontal = null) {
    const supported = Array.from($("target").options)
      .some(option => option.value === target);

    if (
      !supported ||
      ![null, "left", "center", "right"].includes(horizontal)
    ) {
      throw new Error("Unsupported voice target.");
    }

    $("target").value = target;
    voiceSide = horizontal;
    configure();
  }
};