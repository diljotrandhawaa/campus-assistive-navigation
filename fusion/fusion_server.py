#!/usr/bin/env python3
"""GB10 fusion server for the LiDAR Alert iPhone app.

The phone runs ONE ARKit session and sends, for every frame:
    RGB JPEG (upright portrait) + LiDAR depth + confidence + camera pose + intrinsics
This server runs:
    YOLOE (open-vocabulary detection) -> BoT-SORT (track ids) -> sensor fusion
and replies with objects like:  {"track_id": 12, "label": "chair", "distance_m": 2.1, "side": "left"}

Run on the GB10 (conda env indoor-nav has ultralytics, aiohttp, opencv):
    conda activate indoor-nav
    cd ~/indoor-nav/live-yolo-webui          # so the class list in yolo_live/direction.py is reused
    python ~/fusion_server.py --host 0.0.0.0 --port 8092

App: Settings -> server  ws://<GB10 IP>:8092/ws   (e.g. ws://10.36.35.154:8092/ws or ws://100.69.180.2:8092/ws)
Health check from any browser:  http://<GB10 IP>:8092/

Protocol "fusion-1" (one WebSocket):
    server -> {"type":"ready","protocol":"fusion-1","model":...,"classes":[...]}
    app    -> {"type":"configure","confidence":0.35,"revision":1}
    server -> {"type":"configured","revision":1}
    app    -> {"type":"frame","id":N, ...metadata...}  then ONE binary message:
              uint32 LE jpeg_length | jpeg bytes | depth float16 LE (w*h) | confidence uint8 (w*h)
    server -> {"type":"result","id":N,"detector_ms":..,"server_ms":..,"objects":[...]}
              or {"type":"error","id":N,"message":...}
              "target": null, or the find-mode report
              {"label","state":"acquired|tracking|lost_briefly|searching","track_id","distance_m","lateral_m","side"}
    app    -> {"type":"voice_text","text":"find the trash can","request_id":7}   (speech-to-text is on the phone)
    server -> {"type":"voice_intent","request_id":7,"intent":"find","target":"trash can","exclude":[],
               "side":null,"prefer":"nearest","say":"Looking for a trash can.","candidates":[]}
              intents: find, another, cancel, clarify, unsupported, status, help, targets,
                       mute, unmute, repeat, start_app, stop_app
    app    -> {"type":"target","action":"cancel"|"find"|"another","label":...}   (direct find-mode control)
"""
import argparse
import asyncio
import importlib.util
import json
import re
import os
import ssl
import struct
import sys
import time

import numpy as np

PROTOCOL = "fusion-1"
VERSION = "2026-09-25 voice+find"

DEFAULT_CLASSES = [
    "person", "chair", "table", "desk", "door", "stairs", "couch", "bed", "bench", "trash can",
    "backpack", "suitcase", "bicycle", "car", "pole", "potted plant", "box", "cabinet", "shelf",
    "wall", "glass door", "step", "cart", "stroller", "dog", "fire hydrant", "sign", "tv",
]

# ----------------------------------------------------------------------------- fusion math
# Same geometry as DepthSnapshot.swift in the app.
MIN_CONFIDENCE = 1       # ARKit: 0 low, 1 medium, 2 high
FLOOR_MARGIN = 0.10      # points this close above the floor count as floor (m)
MAX_DEPTH = 5.5          # LiDAR is unreliable beyond this (m)
BOX_SHRINK = 0.10        # ignore the outer 10% of each box (background leaks in)
SIDE_THRESHOLD = 0.25    # |lateral| above this is "left"/"right" (m)


class FrameData:
    """One ARKit frame's depth + pose, as sent by the phone."""

    def __init__(self, meta, depth, confidence):
        d = meta["depth"]
        self.w, self.h = int(d["w"]), int(d["h"])
        self.depth = depth.reshape(self.h, self.w)
        self.confidence = confidence.reshape(self.h, self.w)
        self.fx, self.fy, self.cx, self.cy = (float(v) for v in meta["intrinsics"])
        # 16 floats, column-major (simd_float4x4 columns) -> row-major matrix
        self.T = np.asarray(meta["transform"], dtype=np.float64).reshape(4, 4).T
        self.cam_pos = self.T[:3, 3]
        f = np.asarray(meta["forward"], dtype=np.float64)          # horizontal (x, z), unit
        self.forward = f / max(np.linalg.norm(f), 1e-6)
        self.right = np.array([-self.forward[1], self.forward[0]])
        self.floor_y = float(meta["floor_y"])
        self.half_width = float(meta.get("corridor_half_width", 0.35))

    def sensor_rect(self, box):
        """Normalized upright-portrait box [u1,v1,u2,v2] -> depth-map pixel rect.
        Portrait (u, v) maps to sensor (sx, sy) with sx = v, sy = 1 - u."""
        u1, v1, u2, v2 = box
        bw, bh = u2 - u1, v2 - v1
        u1, u2 = u1 + bw * BOX_SHRINK, u2 - bw * BOX_SHRINK
        v1, v2 = v1 + bh * BOX_SHRINK, v2 - bh * BOX_SHRINK
        x0, x1 = int(v1 * self.w), int(v2 * self.w)
        y0, y1 = int((1 - u2) * self.h), int((1 - u1) * self.h)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.w - 1, x1), min(self.h - 1, y1)
        return x0, y0, x1, y1

    def measure(self, box):
        """Returns (distance_m, lateral_m, in_path) or (None, None, False)."""
        x0, y0, x1, y1 = self.sensor_rect(box)
        if x1 < x0 or y1 < y0:
            return None, None, False
        d = self.depth[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
        c = self.confidence[y0:y1 + 1, x0:x1 + 1]
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        ok = (c >= MIN_CONFIDENCE) & np.isfinite(d) & (d > 0.05) & (d < MAX_DEPTH)
        if ok.sum() < 8:
            return None, None, False
        d, xs, ys = d[ok], xs[ok], ys[ok]
        # Depth pixel -> camera space (image y down, camera +Y up, looks along -Z) -> world.
        cam = np.stack([(xs + 0.5 - self.cx) / self.fx * d,
                        -((ys + 0.5 - self.cy) / self.fy * d),
                        -d,
                        np.ones_like(d)])
        world = self.T @ cam
        height = world[1]
        rel = np.stack([world[0] - self.cam_pos[0], world[2] - self.cam_pos[2]])
        ahead = self.forward @ rel
        lateral = self.right @ rel
        keep = (height > self.floor_y + FLOOR_MARGIN) & (ahead > 0.05)
        if keep.sum() < 8:
            return None, None, False
        ahead, lateral = np.sort(ahead[keep]), np.sort(lateral[keep])
        near = float(ahead[len(ahead) // 4])        # 25th percentile = object's near surface
        side = float(lateral[len(lateral) // 2])    # median sideways offset
        in_path = float(np.mean(np.abs(lateral) < self.half_width)) >= 0.10
        return near, side, in_path


class TrackSmoother:
    """Per-track distance smoothing so numbers don't jitter frame to frame."""

    def __init__(self):
        self.tracks = {}  # id -> (distance, lateral, last_seen)

    def update(self, tid, distance, lateral, now):
        if tid is None or distance is None:
            return distance, lateral
        prev = self.tracks.get(tid)
        if prev and now - prev[2] < 1.0 and abs(prev[0] - distance) < 0.8:
            distance = 0.5 * prev[0] + 0.5 * distance
            lateral = 0.5 * prev[1] + 0.5 * lateral
        self.tracks[tid] = (distance, lateral, now)
        for k in [k for k, v in self.tracks.items() if now - v[2] > 5]:
            del self.tracks[k]
        return distance, lateral


def side_text(lateral):
    if lateral is None:
        return None
    if lateral < -SIDE_THRESHOLD:
        return "left"
    if lateral > SIDE_THRESHOLD:
        return "right"
    return "ahead"


def unpack_payload(meta, payload):
    (jpeg_len,) = struct.unpack_from("<I", payload, 0)
    w, h = int(meta["depth"]["w"]), int(meta["depth"]["h"])
    n = w * h
    start = 4 + jpeg_len
    expected = start + n * 2 + n
    if len(payload) != expected:
        raise ValueError(f"payload {len(payload)} bytes, expected {expected}")
    jpeg = payload[4:start]
    depth = np.frombuffer(payload, dtype="<f2", count=n, offset=start).astype(np.float32)
    confidence = np.frombuffer(payload, dtype=np.uint8, count=n, offset=start + n * 2)
    return jpeg, depth, confidence


# ----------------------------------------------------------------------------- detector
class Detector:
    """YOLOE + BoT-SORT. One phone at a time (the tracker keeps state between frames)."""

    def __init__(self, model_path, classes, imgsz, tracker, device):
        from ultralytics import YOLOE
        self.model = YOLOE(model_path)
        try:
            self.model.set_classes(classes, self.model.get_text_pe(classes))
        except (AttributeError, TypeError):
            self.model.set_classes(classes)
        self.model_path = model_path
        self.classes = classes
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device
        # Warm up before any phone connects: the first CUDA run (kernel compilation on
        # the GB10) can take many seconds, long enough for the phone to time out.
        t0 = time.perf_counter()
        blank = np.zeros((640, 480, 3), dtype=np.uint8)
        self.model.track(blank, persist=True, tracker=tracker, imgsz=imgsz, device=device, verbose=False)
        self.reset_tracks()
        print(f"Model warmed up in {time.perf_counter() - t0:.1f} s")

    def reset_tracks(self):
        # Drops the predictor (and its BoT-SORT state); the next call builds a fresh one.
        self.model.predictor = None

    def __call__(self, bgr, confidence):
        r = self.model.track(bgr, persist=True, tracker=self.tracker, conf=confidence,
                             imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return out
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        for xyxyn, cls, conf, tid in zip(boxes.xyxyn.tolist(), boxes.cls.int().tolist(),
                                         boxes.conf.tolist(), ids):
            out.append({"track_id": tid, "label": r.names[cls], "confidence": round(conf, 3),
                        "box": [round(v, 4) for v in xyxyn]})
        return out


def process_frame(detector, smoother, meta, payload, confidence):
    import cv2
    t0 = time.perf_counter()
    jpeg, depth, conf = unpack_payload(meta, payload)
    frame = FrameData(meta, depth, conf)
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("bad JPEG")
    t1 = time.perf_counter()
    detections = detector(bgr, confidence)
    t2 = time.perf_counter()
    now = time.monotonic()
    objects = []
    for det in detections:
        distance, lateral, in_path = frame.measure(det["box"])
        distance, lateral = smoother.update(det["track_id"], distance, lateral, now)
        det.update({
            "distance_m": None if distance is None else round(distance, 3),
            "lateral_m": None if lateral is None else round(lateral, 3),
            "side": side_text(lateral),
            "in_path": bool(in_path),
        })
        objects.append(det)
    objects.sort(key=lambda o: o["distance_m"] if o["distance_m"] is not None else 99)
    return objects, (t2 - t1) * 1000, (time.perf_counter() - t0) * 1000


# ----------------------------------------------------------------------------- find mode
class TargetLock:
    """Find mode: lock onto ONE tracked object of the requested class and report it every frame.

    Same idea as yolo_live/voice_direction.TargetController (lock a BoT-SORT id, tolerate
    short dropouts), but the report is metric (distance + sideways offset from LiDAR)
    instead of "pan camera left/right".
    """

    LOST_GRACE = 3.0  # seconds a locked object may be unseen before we search again

    def __init__(self):
        self.cancel()

    def cancel(self):
        self.label = None
        self.side = None          # "left" / "center" / "right" / None
        self.exclude = set()      # labels to ignore (filled by the future LLM parser)
        self.prefer = "nearest"
        self.skip_ids = set()     # track ids rejected with "another one"
        self.track_id = None
        self.last_seen = None
        self.ever_seen = False

    def find(self, label, side=None, exclude=(), prefer=None):
        self.cancel()
        self.label = label
        self.side = side
        self.exclude = set(exclude or ())
        self.prefer = prefer or "nearest"

    def another(self):
        if self.track_id is not None:
            self.skip_ids.add(self.track_id)
        self.track_id = None
        self.last_seen = None

    def set_side(self, side):
        self.side = side
        self.track_id = None
        self.last_seen = None

    @staticmethod
    def side_of(obj):
        s = obj.get("side")
        if s:
            return "center" if s == "ahead" else s
        u = (obj["box"][0] + obj["box"][2]) / 2
        return "left" if u < 1 / 3 else "right" if u > 2 / 3 else "center"

    def update(self, objects, now):
        if not self.label:
            return None
        base = {"label": self.label, "side_filter": self.side}
        candidates = [o for o in objects
                      if o["label"] == self.label and o["label"] not in self.exclude
                      and o.get("track_id") is not None and o["track_id"] not in self.skip_ids]

        if self.track_id is not None:
            for o in candidates:
                if o["track_id"] == self.track_id:
                    self.last_seen = now
                    o["is_target"] = True
                    return {**base, "state": "tracking", "track_id": o["track_id"],
                            "distance_m": o["distance_m"], "lateral_m": o["lateral_m"], "side": o["side"]}
            if now - self.last_seen <= self.LOST_GRACE:
                return {**base, "state": "lost_briefly", "track_id": self.track_id}
            self.track_id = None  # gone: fall through and search again

        eligible = [o for o in candidates if self.side is None or self.side_of(o) == self.side]
        if not eligible:
            return {**base, "state": "searching", "seen_before": self.ever_seen}
        if self.prefer == "nearest":
            pick = min(eligible, key=lambda o: (o["distance_m"] is None, o["distance_m"] or 0, -o["confidence"]))
        else:
            pick = max(eligible, key=lambda o: o["confidence"])
        self.track_id, self.last_seen, self.ever_seen = pick["track_id"], now, True
        pick["is_target"] = True
        return {**base, "state": "acquired", "track_id": pick["track_id"],
                "distance_m": pick["distance_m"], "lateral_m": pick["lateral_m"], "side": pick["side"]}

    def status_text(self):
        if not self.label:
            return "Not searching for anything."
        where = f" on your {self.side}" if self.side in ("left", "right") else ""
        return f"Looking for {with_article(self.label)}{where}."


# ----------------------------------------------------------------------------- voice
# The phone does speech-to-text on-device and sends TEXT. This turns text into a
# fixed-format intent. Today: the existing yolo_live rules + semantic matcher.
# Later: an LLM parser can fill the same fields (exclude, prefer, ...) with no app change.

VOICE_CONTROLS = {  # yolo_live.voice_intents actions -> app intents
    "cancel_target": "cancel", "stop_all": "cancel", "stop_camera": "stop_app",
    "start_camera": "start_app", "pause": "mute", "resume": "unmute", "repeat": "repeat",
    "status": "status", "help": "help", "targets": "targets",
}
ANOTHER_PHRASES = re.compile(
    r"^(?:(?:no|nope)\s+)?(?:(?:try|find|get|show me)\s+)?"
    r"(?:another(?: one)?|a different one|the other one|not (?:this|that) one|wrong one|"
    r"(?:that|this)(?: one)? is wrong|next one)$")
NEAREST_WORDS = re.compile(r"\b(?:nearest|closest)\b")
HELP_TEXT = ("You can say: find the door, find a chair on my left, another one, cancel, "
             "what's ahead, repeat, mute, or unmute.")


def with_article(label):
    return ("an " if label[:1] in "aeiou" else "a ") + label


def join_or(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]


def find_webui_dir(explicit=None):
    """Folder that contains yolo_live/ (the 8091 browser server's code)."""
    here = os.path.dirname(os.path.abspath(__file__))
    options = [explicit, os.getcwd(), here, os.path.dirname(here),
               os.path.expanduser("~/indoor-nav/live-yolo-webui")]
    for folder in options:
        if folder and os.path.isfile(os.path.join(os.path.expanduser(folder), "yolo_live", "voice_intents.py")):
            return os.path.abspath(os.path.expanduser(folder))
    return None


class VoiceInterpreter:
    """Wraps yolo_live.voice_intents (control phrases + rules + semantic matcher)."""

    def __init__(self, classes, webui_dir=None):
        self.classes = list(classes)
        self._interpret = None
        self._warmup = None
        try:
            folder = find_webui_dir(webui_dir)
            if folder is None:
                raise ImportError("yolo_live/ not found (use --webui-dir ~/indoor-nav/live-yolo-webui)")
            if folder not in sys.path:
                sys.path.insert(0, folder)
            from yolo_live.voice_commands import warmup_command_matcher
            from yolo_live.voice_intents import interpret_command
            self._interpret, self._warmup = interpret_command, warmup_command_matcher
            print(f"Voice: using yolo_live.voice_intents from {folder} (rules + semantic matcher)")
        except Exception as e:  # noqa: BLE001
            print(f"Voice: yolo_live voice matcher unavailable ({e}); exact class names only")

    def warmup(self):
        if self._warmup:
            try:
                self._warmup()
                print("Voice: semantic matcher ready")
            except Exception as e:  # noqa: BLE001
                print(f"Voice: semantic matcher failed to load ({e}); rules still work")

    def interpret(self, text):
        if self._interpret:
            try:
                return self._interpret(text)
            except Exception as e:  # noqa: BLE001
                print(f"Voice: matcher error ({e}); using exact class names")
        return self._fallback(text)

    def _fallback(self, text):
        t = " " + re.sub(r"[^\w\s]", " ", text.lower()) + " "
        found = [c for c in sorted(self.classes, key=len, reverse=True) if f" {c} " in t]
        if not found:
            return {"status": "unsupported_target",
                    "message": "I didn't recognize an object name. Try: find the door."}
        side = next((s for s in ("left", "right", "center") if f" {s} " in t), None)
        return {"status": "resolved", "intent": "select_target", "target": found[0],
                "horizontal": side, "method": "exact"}


def voice_reply(text, interpreter, lock, session, classes):
    """Returns the fixed-format intent reply and updates the target lock."""
    out = {"type": "voice_intent", "transcript": text, "intent": "clarify", "target": None,
           "exclude": [], "side": None, "prefer": None, "say": "", "candidates": [], "method": None}
    clean = " ".join(re.sub(r"[^\w\s]", " ", text.lower().replace("’", "'").replace("'", "")).split())

    if ANOTHER_PHRASES.match(clean):
        if lock.label:
            lock.another()
            return {**out, "intent": "another", "target": lock.label, "method": "rule",
                    "say": f"Looking for another {lock.label}."}
        return {**out, "intent": "clarify", "method": "rule", "say": "What should I look for?"}

    prefer = "nearest" if NEAREST_WORDS.search(clean) else None
    query = NEAREST_WORDS.sub(" ", clean) if prefer else text  # the rules reject "nearest"; LiDAR handles it
    r = interpreter.interpret(query)
    out["method"] = r.get("method")
    status, intent = r.get("status"), r.get("intent")

    def start_find(target, side):
        if target not in classes:
            return {**out, "intent": "unsupported", "say": f"I can't detect {with_article(target)} yet."}
        side = "center" if side in ("centre", "middle") else side
        lock.find(target, side=side, prefer=prefer)
        session["candidates"] = []
        where = f" on your {side}" if side in ("left", "right") else " in front of you" if side == "center" else ""
        near = "the nearest " + target if prefer else with_article(target)
        return {**out, "intent": "find", "target": target, "side": side, "prefer": lock.prefer,
                "say": f"Looking for {near}{where}."}

    if status == "resolved":
        if intent == "select_target":
            return start_find(r.get("target"), r.get("horizontal"))
        if intent == "clarify_side":
            side = r.get("horizontal")
            if lock.label:
                lock.set_side(side)
                return {**out, "intent": "find", "target": lock.label, "side": side,
                        "say": f"Looking for the {lock.label} on your {side}."}
            if len(session.get("candidates", [])) == 1:
                return start_find(session["candidates"][0], side)
            return {**out, "say": "Which object should I look for?"}
        if intent == "clarify_choice":
            options = session.get("candidates", [])
            i = r.get("choice", 0)
            if i < len(options):
                return start_find(options[i], None)
            return {**out, "say": "Which object should I look for?"}
        action = VOICE_CONTROLS.get(intent)
        if action == "cancel":
            had = lock.label
            lock.cancel()
            return {**out, "intent": "cancel", "say": f"Stopped looking for the {had}." if had else "Cancelled."}
        if action == "status":
            return {**out, "intent": "status", "target": lock.label, "say": lock.status_text()}
        if action == "help":
            return {**out, "intent": "help", "say": HELP_TEXT}
        if action == "targets":
            simple = [c for c in classes if " " not in c][:12]
            return {**out, "intent": "targets", "candidates": list(classes),
                    "say": f"I can find {len(classes)} kinds of things, including {join_or(simple)}."}
        if action:
            return {**out, "intent": action}  # mute / unmute / repeat / start_app / stop_app: the app acts
        return {**out, "intent": "unsupported",
                "say": "That command only works on the web page for now."}

    # needs_clarification / unsupported_target / no_speech
    raw = r.get("candidates") or []
    options = [c["target"] if isinstance(c, dict) else c for c in raw]
    options = [c for c in options if c in classes] if status == "needs_clarification" else []
    session["candidates"] = options
    say = r.get("message") or "Sorry, I didn't understand."
    if options and not any(o in say for o in options):
        say += f" Did you mean {join_or(options)}?"
    return {**out, "intent": "clarify" if status == "needs_clarification" else "unsupported",
            "candidates": options, "say": say}


# ----------------------------------------------------------------------------- server
def load_classes(args):
    if args.classes:
        return [c.strip() for c in args.classes.split(",") if c.strip()]
    folder = find_webui_dir(args.webui_dir) or os.getcwd()
    path = os.path.join(folder, "yolo_live", "direction.py")
    if os.path.exists(path):
        try:
            spec = importlib.util.spec_from_file_location("direction", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            classes = list(getattr(mod, "CLASSES"))
            print(f"Using {len(classes)} classes from {path}")
            return classes
        except Exception as e:  # noqa: BLE001
            print(f"Could not read CLASSES from {path}: {e}")
    print(f"Using {len(DEFAULT_CLASSES)} default classes (override with --classes)")
    return DEFAULT_CLASSES


def build_app(detector, interpreter=None, webui_dir=None):
    from concurrent.futures import ThreadPoolExecutor

    from aiohttp import WSMsgType, web

    state = {"client": None}
    gpu = asyncio.Lock()
    interpreter = interpreter or VoiceInterpreter(detector.classes, webui_dir)
    voice_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice")

    async def warm_voice(_app):
        asyncio.get_running_loop().run_in_executor(voice_pool, interpreter.warmup)

    async def health(_request):
        return web.Response(text=f"LiDAR Alert fusion server ({PROTOCOL}, {VERSION}) OK. "
                                 f"model={detector.model_path} classes={len(detector.classes)} "
                                 f"phone={'connected' if state['client'] else 'none'}\n")

    async def safe_send(ws, data):
        if ws.closed:
            return False
        try:
            await ws.send_json(data)
            return True
        except (ConnectionResetError, RuntimeError) as e:  # aiohttp ClientConnectionResetError is a ConnectionResetError
            print(f"Phone went away while sending ({type(e).__name__})")
            return False

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=8_000_000, heartbeat=10)
        await ws.prepare(request)
        peer = request.remote
        if state["client"] is not None and not state["client"].closed:
            # A new phone connection replaces the old one (e.g. app restarted).
            await state["client"].close()
        state["client"] = ws
        detector.reset_tracks()
        smoother = TrackSmoother()
        lock = TargetLock()
        session = {"candidates": []}
        confidence = 0.35
        pending_meta = None
        print(f"Phone connected: {peer}")
        await ws.send_json({"type": "ready", "protocol": PROTOCOL, "model": detector.model_path,
                            "classes": detector.classes, "imgsz": detector.imgsz})
        loop = asyncio.get_running_loop()
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        m = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    kind = m.get("type")
                    if kind == "configure":
                        confidence = min(0.9, max(0.05, float(m.get("confidence", confidence))))
                        await safe_send(ws, {"type": "configured", "revision": m.get("revision", 0)})
                    elif kind == "frame":
                        pending_meta = m
                    elif kind == "voice_text":
                        text = str(m.get("text", ""))[:400]
                        try:
                            reply = await loop.run_in_executor(
                                voice_pool, voice_reply, text, interpreter, lock, session, detector.classes)
                        except Exception as e:  # noqa: BLE001
                            reply = {"type": "voice_intent", "intent": "unsupported", "transcript": text,
                                     "target": None, "exclude": [], "side": None, "prefer": None,
                                     "candidates": [], "method": None,
                                     "say": "Sorry, voice commands failed on the server."}
                            print(f"Voice error: {e}")
                        reply["request_id"] = m.get("request_id")
                        print(f"Voice: {text!r} -> {reply['intent']} {reply.get('target') or ''}")
                        await safe_send(ws, reply)
                    elif kind == "target":
                        # Direct control from the app (local "cancel", arrival, or a tapped choice).
                        action = m.get("action")
                        if action == "cancel":
                            lock.cancel()
                        elif action == "find" and m.get("label") in detector.classes:
                            lock.find(m["label"], side=m.get("side"), exclude=m.get("exclude") or (),
                                      prefer=m.get("prefer"))
                        elif action == "another":
                            lock.another()
                elif msg.type == WSMsgType.BINARY:
                    meta, pending_meta = pending_meta, None
                    if meta is None:
                        continue
                    fid = meta.get("id")
                    try:
                        async with gpu:
                            objects, det_ms, srv_ms = await loop.run_in_executor(
                                None, process_frame, detector, smoother, meta, msg.data, confidence)
                    except Exception as e:  # noqa: BLE001
                        print(f"Frame {fid} failed: {e}")
                        await safe_send(ws, {"type": "error", "id": fid, "message": str(e)[:200]})
                        continue
                    target = lock.update(objects, time.monotonic())
                    await safe_send(ws, {"type": "result", "id": fid, "objects": objects, "target": target,
                                         "detector_ms": round(det_ms, 1), "server_ms": round(srv_ms, 1)})
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            if state["client"] is ws:
                state["client"] = None
            print(f"Phone disconnected: {peer}")
        return ws

    app = web.Application()
    app.on_startup.append(warm_voice)
    app.router.add_get("/", health)
    app.router.add_get("/ws", ws_handler)
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--model", default="yoloe-11l-seg.pt", help="YOLOE weights (downloaded if missing)")
    p.add_argument("--classes", help="comma-separated class names (default: yolo_live/direction.py CLASSES)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tracker", default="botsort.yaml")
    p.add_argument("--device", default=None, help="e.g. 0 or cuda:0 (default: auto)")
    p.add_argument("--webui-dir", help="folder containing yolo_live/ (default: auto-detect, "
                                       "e.g. ~/indoor-nav/live-yolo-webui)")
    p.add_argument("--cert", help="TLS certificate (then use wss:// in the app)")
    p.add_argument("--key", help="TLS private key")
    args = p.parse_args()

    from aiohttp import web
    print(f"LiDAR Alert fusion server, version {VERSION}")
    classes = load_classes(args)
    print(f"Loading {args.model} ...")
    detector = Detector(args.model, classes, args.imgsz, args.tracker, args.device)
    ssl_ctx = None
    if args.cert and args.key:
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(args.cert, args.key)
    scheme = "wss" if ssl_ctx else "ws"
    print(f"Ready. App server address: {scheme}://<this machine's IP>:{args.port}/ws")
    web.run_app(build_app(detector, webui_dir=args.webui_dir), host=args.host, port=args.port, ssl_context=ssl_ctx, print=None)


if __name__ == "__main__":
    sys.exit(main())
