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
              "target": null, or the find-mode report; "place": room|restroom|hallway|lobby|stairwell|unknown
              {"label","state":"acquired|tracking|lost_briefly|searching","track_id","distance_m","lateral_m",
               "side","bearing_deg"}   (bearing works beyond LiDAR range; + = right)
    app    -> {"type":"voice_text","text":"find the trash can","request_id":7}   (speech-to-text is on the phone)
    server -> {"type":"voice_intent","request_id":7,"intent":"find","target":"trash can","exclude":[],
               "side":null,"prefer":"nearest","say":"Looking for a trash can.","candidates":[]}
              intents: find, another, cancel, clarify, unsupported, status, help, targets,
                       mute, unmute, repeat, start_app, stop_app
    app    -> {"type":"target","action":"cancel"|"find"|"another","label":...}   (direct find-mode control)
    server -> {"type":"text_result","say":"Sign, 8 feet, slightly left: Room 204.","groups":[...]}
              (after "read the sign"; OCR of the next frame with distance/direction per sign)
    app    -> {"type":"blocked","request_id":N,"target":{...},"layout":{...},"obstacle":...}   (way_around.py)
    server -> {"type":"way","request_id":N,"status":"default|processing|answer","say":...}
    Places ("find the restroom", "take me to room 204", "find the exit") use the same find mode:
    candidates are sign/door boxes, OCR reads each tracked candidate once, text-matching wins.
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

from text_utils import (clean_text, count_phrase, direction_words, join_or, natural_join, side_text, spoken_feet,  # noqa: F401
                        target_phrase, text_matches, text_words, with_article)
from scene_memory import SceneMemory  # noqa: F401
import place_vlm  # where the user is (room / hallway / ...), from the camera model (VLM)
from geometry import BOX_SHRINK, FLOOR_MARGIN, MAX_DEPTH, MIN_CONFIDENCE, FrameData, heading_deg  # noqa: F401
from place_search import PlaceSearch
from sign_reader import (BARE_NUMBER, LABELED_NUMBER, SignReading, answer_verify, answer_which,  # noqa: F401
                         describe_read, group_text_items, where_text)
from target_lock import TargetLock  # noqa: F401  (find mode: lock, world anchor, report)
from exit_planner import (BUILDING_EXIT, EXIT_REQUEST, EXIT_SAY, ROOM_EXIT, exit_spec,  # noqa: F401
                          plan_exit)
import way_around  # way around an obstacle on the way to the target, in a room (LLM + checks)

try:  # the LLM part is optional: without intent_llm.py the server runs rules-only
    from intent_llm import LLMIntent, clean_llm
except ImportError:
    LLMIntent = clean_llm = None

PROTOCOL = "fusion-1"
VERSION = "2026-09-26 voice+find+cameratracker+bearing+signs-ocr+llm(intent_llm.py)+exit-planner+place-search+anchored-target+way-around-llm+place-vlm+refind"

DEFAULT_CLASSES = [
    "person", "chair", "table", "desk", "door", "stairs", "couch", "bed", "bench", "trash can",
    "backpack", "suitcase", "bicycle", "car", "pole", "potted plant", "box", "cabinet", "shelf",
    "wall", "glass door", "step", "cart", "stroller", "dog", "fire hydrant", "sign", "tv",
]

# (frame geometry: geometry.py)


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


# ----------------------------------------------------------------------------- places + sign text
# A "find" target is a spec: which YOLOE classes are candidates, which of them count on
# their own ("specific"), and which words OCR must read on a candidate ("keywords").
#   chair       -> classes {chair}, specific {chair}, no keywords
#   restroom    -> classes {restroom sign, sign, door, toilet}, keywords restroom/men/women/...
#   room 204    -> classes {sign, door}, keyword "204"  (needs OCR)
EXTRA_CLASSES = ["restroom sign", "exit sign"]  # added to the YOLOE prompts at startup

PLACES = {
    "restroom": {"classes": ["restroom sign", "sign", "door", "toilet"],
                 "specific": ["restroom sign", "toilet"],
                 "keywords": ["restroom", "restrooms", "bathroom", "bathrooms", "toilet", "toilets", "wc",
                              "men", "women", "mens", "womens", "ladies", "gents", "lavatory", "washroom",
                              "all gender", "unisex", "family restroom"]},
    "exit": {"classes": ["exit sign", "sign", "door"], "specific": ["exit sign"], "keywords": ["exit"]},
}
PLACE_PATTERNS = [
    (re.compile(r"\b(?:rest ?rooms?|bath ?rooms?|wash ?rooms?|toilets?|lavator(?:y|ies)|loo|wc|w c|"
                r"mens room|womens room|men s room|women s room|ladies room|gents)\b"), "restroom"),
    (re.compile(r"\b(?:exit|way out)\b"), "exit"),
]
ROOM_RE = re.compile(r"\broom (?:number )?(\d{1,4}[a-z]?)\b")
PLACE_BLOCK = re.compile(r"\b(?:stop|cancel|not|dont|never|without|avoid)\b")


def parse_place(clean):
    """clean = lowercase words. Returns "restroom", "exit", "room 204" or None."""
    if PLACE_BLOCK.search(clean):
        return None
    m = ROOM_RE.search(clean)
    if m:
        return f"room {m.group(1)}"
    for pattern, name in PLACE_PATTERNS:
        if pattern.search(clean):
            return name
    return None


def resolve_target(label, classes):
    """Target name -> spec dict, or None if this server can't look for it."""
    classes = set(classes)
    if label == "exit":   # re-sent by the phone after a reconnect: sign first, then the door by it
        return exit_spec("hallway", classes)
    if label in PLACES:
        p = PLACES[label]
        cl = {c for c in p["classes"] if c in classes}
        if not cl:
            return None
        return {"label": label, "classes": cl, "keywords": p["keywords"],
                "specific": {c for c in p["specific"] if c in classes}}
    m = re.fullmatch(r"room (\d{1,4}[a-z]?)", label or "")
    if m:
        cl = {c for c in ("sign", "door") if c in classes}
        return {"label": label, "classes": cl, "keywords": [m.group(1)], "specific": set()} if cl else None
    if label in classes:
        return {"label": label, "classes": {label}, "keywords": [], "specific": {label}}
    return None


# (reading signs aloud: sign_reader.py)


# ----------------------------------------------------------------------------- detector
class Detector:
    """Shared YOLOE model + per-phone tracking of the find-mode target only.

    Preferred: yolo_live.tracking.CameraTracker (the 8091 server's BoT-SORT with ReID,
    per-session state, time-based expiry). It only tracks the requested label, so when
    find mode is off nothing is tracked at all.
    Fallback (yolo_live not found, or the ReID file is missing): Ultralytics' built-in
    BoT-SORT via model.track(), keeping ids for the target label only.
    """

    def __init__(self, model_path, classes, imgsz, tracker, device, webui_dir=None,
                 ocr=True, ocr_confidence=0.6):
        from ultralytics import YOLOE
        self.model = YOLOE(model_path)
        try:
            self.model.set_classes(classes, self.model.get_text_pe(classes))
        except (AttributeError, TypeError):
            self.model.set_classes(classes)
        self.model_path = model_path
        self.classes = classes          # base classes (what the phone is told)
        self.active_classes = list(classes)
        self._pending_classes = None    # set_prompts() -> applied before the next frame
        self.imgsz = imgsz
        self.tracker = tracker          # built-in fallback config, e.g. botsort.yaml
        self.device = device
        self.camera_tracker_cls = None

        # Warm up before any phone connects: the first CUDA run (kernel compilation on
        # the GB10) can take many seconds, long enough for the phone to time out.
        t0 = time.perf_counter()
        blank = np.zeros((640, 480, 3), dtype=np.uint8)
        self.model.predict(blank, imgsz=imgsz, device=device, conf=0.25, verbose=False)
        print(f"Model warmed up in {time.perf_counter() - t0:.1f} s")

        self.ocr = self._load_ocr(webui_dir, ocr_confidence) if ocr else None
        if not ocr:
            print("OCR: off (--no-ocr)")

        reason = self._load_camera_tracker(webui_dir, blank)
        if self.camera_tracker_cls:
            print("Tracking: yolo_live CameraTracker (BoT-SORT + ReID), find-mode target only")
        else:
            print(f"Tracking: Ultralytics built-in BoT-SORT ({reason})")

    def _load_camera_tracker(self, webui_dir, blank):
        folder = find_webui_dir(webui_dir)
        if folder is None:
            return "yolo_live/ not found; use --webui-dir"
        if not os.path.exists("yolo26n-reid.onnx"):
            return ("yolo26n-reid.onnx is not in the current folder; "
                    f"run the server from {folder} to use CameraTracker")
        try:
            if folder not in sys.path:
                sys.path.insert(0, folder)
            from yolo_live.tracking import CameraTracker
            from ultralytics.trackers.bot_sort import BOTSORT  # noqa: F401  (fails early if 'lap' is missing)
            # Warm up tracker + ReID model so the first "find" isn't slow.
            t0 = time.perf_counter()
            r = self.model.predict(blank, imgsz=self.imgsz, device=self.device, conf=0.25, verbose=False)[0]
            CameraTracker().update(r.boxes.cpu().numpy(), blank, r.names, self.classes[0], 0.35, time.monotonic())
            print(f"Tracker warmed up in {time.perf_counter() - t0:.1f} s")
            self.camera_tracker_cls = CameraTracker
            return None
        except Exception as e:  # noqa: BLE001
            return f"CameraTracker unavailable: {e}"

    def _load_ocr(self, webui_dir, min_score):
        """The 8091 server's PP-OCRv5 reader (yolo_live/ocr.py), on the same GPU."""
        folder = find_webui_dir(webui_dir)
        if folder is None:
            print("OCR: yolo_live/ not found (use --webui-dir); sign reading disabled")
            return None
        try:
            if folder not in sys.path:
                sys.path.insert(0, folder)
            from yolo_live.ocr import OCRBackend
            t0 = time.perf_counter()
            backend = OCRBackend(self.device if self.device is not None else 0, min_score, max_regions=8)
            backend.load()  # loads and warms up both OCR models
            print(f"OCR: yolo_live PP-OCRv5 ready in {time.perf_counter() - t0:.1f} s")
            return backend
        except Exception as e:  # noqa: BLE001
            print(f"OCR: unavailable ({e}); sign reading disabled")
            return None

    def read_boxes(self, bgr, boxes):
        """OCR inside each normalized box (a sign or door crop). Returns one string per box
        ("" = nothing readable, None = too small to try)."""
        import cv2
        h, w = bgr.shape[:2]
        out = []
        for x1, y1, x2, y2 in boxes:
            pw, ph = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            a, b = int(max(0, (x1 - pw) * w)), int(max(0, (y1 - ph) * h))
            c, d = int(min(w, (x2 + pw) * w)), int(min(h, (y2 + ph) * h))
            crop = bgr[b:d, a:c]
            if min(crop.shape[:2]) < 16:
                out.append(None)
                continue
            if crop.shape[0] < 96:  # small text reads better enlarged
                f = 96 / crop.shape[0]
                crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            items = self.ocr.infer(np.ascontiguousarray(crop), []).get("items", [])
            out.append(" ".join(i["text"] for i in items))
        return out

    def read_full(self, bgr, objects):
        """OCR on the whole frame (for "read the sign"); links text to nearby doors/signs."""
        return self.ocr.infer(bgr, objects).get("items", [])

    def set_prompts(self, classes):
        """Change YOLOE's text prompts (a place search adds sign phrases, then removes them).
        Applied on the GPU thread right before the next frame, never during one."""
        self._pending_classes = list(classes)

    def _apply_prompts(self):
        classes, self._pending_classes = self._pending_classes, None
        if classes is None or classes == getattr(self, "active_classes", self.classes):
            return
        t0 = time.perf_counter()
        try:
            self.model.set_classes(classes, self.model.get_text_pe(classes))
        except (AttributeError, TypeError):
            self.model.set_classes(classes)
        self.model.predictor = None  # rebuild with the new names
        self.active_classes = classes
        print(f"Prompts: YOLOE now has {len(classes)} classes ({(time.perf_counter() - t0) * 1000:.0f} ms)")

    def new_session(self):
        """Tracking state for one phone connection."""
        if self.camera_tracker_cls:
            return {"kind": "camera", "tracker": self.camera_tracker_cls()}
        self.model.predictor = None  # fresh built-in tracker state
        return {"kind": "builtin", "target": None}

    def __call__(self, bgr, confidence, session, target):
        """Returns (detections, tracking_info). `target` is a set of class names (or None);
        only those detections get track ids."""
        if getattr(self, "_pending_classes", None) is not None:
            self._apply_prompts()
        height, width = bgr.shape[:2]
        if session["kind"] == "camera":
            tracker = session["tracker"]
            # While tracking, detect down to the tracker's low threshold so a confidence dip
            # doesn't lose the lock (same as yolo_live/engine.py); others still need `confidence`.
            conf = min(confidence, tracker.low_confidence) if target else confidence
            r = self.model.predict(bgr, imgsz=self.imgsz, device=self.device, conf=conf,
                                   max_det=60, verbose=False)[0]
            boxes = r.boxes.cpu().numpy()
            assignments, tracking = {}, None
            if target:
                # CameraTracker follows ONE label. Give every target class the same name so it
                # tracks e.g. signs and doors together; the key changes (-> fresh tracker) when
                # the target set changes.
                key = "target:" + "|".join(sorted(target))
                names = {i: (key if n in target else n) for i, n in r.names.items()}
                try:
                    assignments, tracking = tracker.update(boxes, bgr, names, key, confidence, time.monotonic())
                except Exception as e:  # noqa: BLE001
                    print(f"Tracker error (frame continues without ids): {e}")
            out = []
            for index, row in enumerate(boxes.data.tolist()):
                x1, y1, x2, y2, score, cls = row[:6]
                tid = assignments.get(index)
                if score < confidence and tid is None:
                    continue
                out.append({"track_id": tid, "label": r.names[int(cls)], "confidence": round(score, 3),
                            "box": [round(max(0, min(1, v)), 4)
                                    for v in (x1 / width, y1 / height, x2 / width, y2 / height)]})
            return out, tracking

        # Built-in fallback: model.track keeps its state inside the model's predictor.
        if target != session["target"]:
            self.model.predictor = None
            session["target"] = target
        r = self.model.track(bgr, persist=True, tracker=self.tracker, conf=confidence,
                             imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return out, None
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        for xyxyn, cls, conf, tid in zip(boxes.xyxyn.tolist(), boxes.cls.int().tolist(),
                                         boxes.conf.tolist(), ids):
            label = r.names[cls]
            out.append({"track_id": tid if target and label in target else None, "label": label,
                        "confidence": round(conf, 3), "box": [round(v, 4) for v in xyxyn]})
        return out, None


def process_frame(detector, sess, meta, payload, confidence):
    """One phone frame -> objects (+ tracking info, + OCR text for "read the sign")."""
    import cv2
    t0 = time.perf_counter()
    jpeg, depth, conf = unpack_payload(meta, payload)
    frame = FrameData(meta, depth, conf)
    sess["last_frame"] = frame          # for the place search planner (floor map, headings)
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("bad JPEG")
    lock = sess["lock"]
    spec = lock.spec
    t1 = time.perf_counter()
    detections, tracking = detector(bgr, confidence, sess["tracking"], spec["classes"] if spec else None)
    t2 = time.perf_counter()
    objects = []
    for det in detections:
        distance, lateral, in_path = frame.measure(det["box"])
        bearing = frame.bearing(det["box"])
        det.update({
            "bearing_deg": None if bearing is None else round(bearing, 1),
            "distance_m": None if distance is None else round(distance, 3),
            "lateral_m": None if lateral is None else round(lateral, 3),
            "side": side_text(lateral),
            "in_path": bool(in_path),
        })
        objects.append(det)
    objects.sort(key=lambda o: o["distance_m"] if o["distance_m"] is not None else 99)

    now = time.monotonic()
    if sess.get("scene") is not None:
        sess["scene"].add_frame(now, objects, frame.space_hint(), frame=frame)
    place = sess.get("place")
    if place is not None:   # the VLM sees a few recent frames, one per direction, every few seconds
        place.add_frame(jpeg, heading_deg(frame.forward), now)
        place.tick(now, lambda: place_vlm.scene_hints(sess.get("scene"), now))
    if getattr(detector, "ocr", None) is not None and spec and spec["keywords"]:
        read_target_text(detector, sess, bgr, objects, tracking, now)

    read = None
    if sess.get("read_pending"):
        # "Read the sign": keep looking over several frames while the user turns (sign_reader.py).
        if getattr(detector, "ocr", None) is None:
            read = {"say": "Reading signs isn't available on the server.", "groups": [], "done": True}
        else:
            if sess.get("reading") is None:
                sess["reading"] = SignReading(sess.get("read_mode"), now)

            def read_groups():
                groups = group_text_items(detector.read_full(bgr, objects))
                for g in groups:
                    d, _, _ = frame.measure(g["box"])
                    g["distance_m"] = None if d is None else round(d, 2)
                    b = frame.bearing(g["box"])
                    g["bearing_deg"] = None if b is None else round(b, 1)
                groups.sort(key=lambda g: g["distance_m"] if g["distance_m"] is not None else 99)
                if sess.get("scene") is not None:
                    for g in groups:
                        sess["scene"].add_text(now, " ".join(g["texts"]))
                return groups

            read = sess["reading"].step(now, read_groups)
        if read is not None and read.get("done"):
            sess["read_pending"] = False
            sess["reading"] = None
            sess["read_mode"] = None
    return objects, tracking, read, (t2 - t1) * 1000, (time.perf_counter() - t0) * 1000


OCR_MIN_INTERVAL = 0.5   # seconds between OCR passes during a text-based find
OCR_PER_PASS = 2         # candidate boxes read per pass
OCR_RETRIES = 6          # an unreadable (far/blurry) sign is retried this many times


def read_target_text(detector, sess, bgr, objects, tracking, now):
    """Reads candidate signs/doors ONCE per track id and remembers the text."""
    cache = sess["ocr_cache"]
    gen = (tracking or {}).get("generation")
    if gen != sess.get("ocr_generation"):  # tracker rebuilt: old ids mean nothing
        cache.clear()
        sess["ocr_generation"] = gen
    classes = sess["lock"].spec["classes"]
    candidates = [o for o in objects if o["label"] in classes and o.get("track_id") is not None]
    for o in candidates:
        entry = cache.get(o["track_id"])
        if entry and entry["text"]:
            o["text"] = entry["text"]
    if now - sess.get("last_ocr", 0.0) < OCR_MIN_INTERVAL:
        return
    todo = [o for o in candidates
            if o["track_id"] not in cache
            or (not cache[o["track_id"]]["text"] and cache[o["track_id"]]["tries"] < OCR_RETRIES
                and now - cache[o["track_id"]]["last"] > 1.0)]
    # Nearest / biggest first
    todo.sort(key=lambda o: (o["distance_m"] is None, o["distance_m"] or 0,
                             -(o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])))
    todo = todo[:OCR_PER_PASS]
    if not todo:
        return
    sess["last_ocr"] = now
    try:
        texts = detector.read_boxes(bgr, [o["box"] for o in todo])
    except Exception as e:  # noqa: BLE001
        print(f"OCR error: {e}")
        return
    for o, text in zip(todo, texts):
        if text is None:  # too small to read yet; try again when closer
            continue
        entry = cache.setdefault(o["track_id"], {"text": "", "tries": 0, "last": 0.0})
        entry["tries"] += 1
        entry["last"] = now
        if text:
            entry["text"] = text
            o["text"] = text
            if sess.get("scene") is not None:
                sess["scene"].add_text(now, text)
            print(f"OCR: #{o['track_id']} {o['label']} reads {text!r}")


# ----------------------------------------------------------------------------- find mode
# (scene memory + place rules: scene_memory.py; "find an exit": exit_planner.py)


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
CANCEL_REQUEST = re.compile(r"^(?:please )?(?:stop|cancel|quit|end)(?: (?:looking|searching|finding|the search|search|it))?\b(?! (?:app|camera))")
VERIFY_QUESTION = re.compile(r"^(?:is (?:this|that|it|here)|am i (?:at|in|by)|are we (?:at|in|by))\b")
READ_REQUEST = re.compile(r"^(?:please )?(?:can you )?(?:read|what does (?:it|this|that|the sign) say)\b")
HELP_TEXT = ("You can say: find the door, find the restroom, take me to room 204, read the sign, "
             "another one, cancel, what's ahead, repeat, mute, or unmute.")


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


# ----------------------------------------------------------------------------- LLM intent parser (glue)
# The model itself (prompt, JSON schema, Ollama client, validation) lives in intent_llm.py.
# Here: turning its validated answer into find mode / sign reading.

def apply_llm(d, out, lock, session, classes):
    """Turns validated LLM output into the same reply format as the rules."""
    out = {**out, "method": "llm"}
    i = d["i"]
    if i == "cancel":
        had = lock.label
        lock.cancel()
        return {**out, "intent": "cancel", "say": f"Stopped looking for {target_phrase(had)}." if had else "Cancelled."}
    if i == "another":
        if not lock.label:
            return {**out, "intent": "clarify", "say": "What should I look for?"}
        lock.another()
        return {**out, "intent": "another", "target": lock.label, "say": f"Looking for another {lock.label}."}
    if i in ("read", "verify", "which"):
        if not session.get("ocr"):
            return {**out, "intent": "unsupported", "say": "Reading signs isn't available on the server."}
        if i == "verify" and not d["k"]:
            return {**out, "intent": "clarify", "say": "What should the sign say?"}
        session["read_pending"] = True
        session["read_mode"] = ({"kind": "verify", "keywords": d["k"]} if i == "verify"
                                else {"kind": "which", "room_kind": d["r"]} if i == "which" else None)
        return {**out, "intent": "read", "say": ""}
    if i != "find":
        return {**out, "intent": "unsupported", "say": "Sorry, I can help you find things and read signs."}

    t, k = d["t"], d["k"]
    if k and (t in (None, "sign", "door") or t not in classes):
        # A place identified by its sign text: read signs and doors.
        cl = {c for c in ("sign", "door") if c in classes}
        spec = {"label": " ".join(k), "classes": cl, "keywords": k, "specific": set()}
        if not session.get("ocr"):
            return {**out, "intent": "unsupported", "say": "Finding that needs sign reading, which isn't available."}
        say = f"Looking for a sign that says {' '.join(k)}."
    elif t:
        spec = resolve_target(t, classes)
        if spec is None:
            return {**out, "intent": "unsupported", "say": f"I can't detect {with_article(t)} yet."}
        if k:  # e.g. "the door that says 204": this class AND matching text
            spec = {**spec, "keywords": k, "specific": set()}
        say = f"Looking for {'the nearest ' + t if d['p'] else with_article(t)}"
        if d["s"] in ("left", "right"):
            say += f" on your {d['s']}"
        if d["n"]:
            say += f" near the {d['n']}"
        if k:
            say += f" that says {' '.join(k)}"
        if d["x"]:
            say += f", not the {join_or(d['x'])}"
        say += "."
    else:
        return {**out, "intent": "clarify", "say": "What should I look for?"}
    if d["n"]:
        spec = {**spec, "near": {"classes": {d["n"]}, "within_m": 1.5}}
    lock.find(spec, side=d["s"], exclude=d["x"], prefer=d["p"])
    session["candidates"] = []
    return {**out, "intent": "find", "target": spec["label"], "exclude": d["x"], "side": d["s"],
            "prefer": lock.prefer, "say": say}


def voice_reply(text, interpreter, lock, session, classes, llm=None):
    """Rules first (instant); the LLM only when the rules can't handle the request."""
    clean = clean_text(text)
    if (EXIT_REQUEST.search(clean) and not PLACE_BLOCK.search(clean)
            and not READ_REQUEST.match(clean) and not VERIFY_QUESTION.match(clean)):
        return plan_exit(text, clean, lock, session, classes, llm)
    reply = rules_reply(text, interpreter, lock, session, classes)
    if llm is None or reply["intent"] not in ("clarify", "unsupported"):
        return reply
    try:
        parsed = clean_llm(llm.parse(text), classes)
    except Exception as e:  # noqa: BLE001  (timeout, Ollama down, bad JSON)
        print(f"LLM: failed ({e}); using the rules' answer")
        return reply
    if parsed is None:
        return reply
    base = {"type": "voice_intent", "transcript": text, "intent": "clarify", "target": None,
            "exclude": [], "side": None, "prefer": None, "say": "", "candidates": [], "method": "llm"}
    return apply_llm(parsed, base, lock, session, classes)


def voice_turn(text, interpreter, lock, session, classes, llm=None):
    """One voice command: a yes/no for the place search first, then the normal rules / LLM;
    a place find then starts the search planner (place_search.py)."""
    search = session.get("search")
    if search is None:
        return voice_reply(text, interpreter, lock, session, classes, llm)
    clean = clean_text(text)
    session["reading"] = None                      # a new request restarts / ends any sign reading
    reply = search.answer(text, clean)
    if reply is None:
        reply = search.after_reply(text, voice_reply(text, interpreter, lock, session, classes, llm))
    if reply.get("intent") != "read":
        session["read_pending"] = False
    return reply


def rules_reply(text, interpreter, lock, session, classes):
    """Returns the fixed-format intent reply and updates the target lock."""
    out = {"type": "voice_intent", "transcript": text, "intent": "clarify", "target": None,
           "exclude": [], "side": None, "prefer": None, "say": "", "candidates": [], "method": None}
    clean = clean_text(text)

    if ANOTHER_PHRASES.match(clean):
        if lock.label:
            lock.another()
            return {**out, "intent": "another", "target": lock.label, "method": "rule",
                    "say": f"Looking for another {lock.label}."}
        return {**out, "intent": "clarify", "method": "rule", "say": "What should I look for?"}

    # Places are found through their signs (YOLOE sign classes + OCR text), same find mode.
    # "stop looking for the restroom", "cancel the search"
    if CANCEL_REQUEST.match(clean):
        had = lock.label
        lock.cancel()
        return {**out, "intent": "cancel", "method": "rule",
                "say": f"Stopped looking for {target_phrase(had)}." if had else "Cancelled."}

    # "read the sign", "read the restroom sign", "what does it say"
    if READ_REQUEST.match(clean):
        if not session.get("ocr"):
            return {**out, "intent": "unsupported", "method": "rule",
                    "say": "Reading signs isn't available on the server."}
        session["read_pending"] = True   # read while the user turns (sign_reader.py); the answer comes as "text_result"
        session["read_mode"] = None
        return {**out, "intent": "read", "method": "rule", "say": ""}

    # "is this the restroom?", "am I at room 204?" -> check the sign instead of starting a search
    if VERIFY_QUESTION.match(clean):
        place = parse_place(clean)
        spec = resolve_target(place, classes) if place else None
        if spec and spec["keywords"] and session.get("ocr"):
            session["read_pending"] = True
            session["read_mode"] = {"kind": "verify", "keywords": spec["keywords"]}
            return {**out, "intent": "read", "method": "rule", "say": ""}
        return {**out, "intent": "unsupported", "say": "I can't check that yet."}  # -> LLM, if enabled

    place = parse_place(clean)
    if place:
        spec = resolve_target(place, classes)
        if spec is None:
            return {**out, "intent": "unsupported", "say": f"I can't look for {target_phrase(place)} with this setup."}
        if not spec["specific"] and not session.get("ocr"):
            return {**out, "intent": "unsupported",
                    "say": f"Finding {target_phrase(place)} needs sign reading, which isn't available on the server."}
        lock.find(spec)
        session["candidates"] = []
        how = ("I'll read the signs by the doors." if place.startswith("room ")
               else "I'll check signs and doors." if session.get("ocr") else "I'll look for its sign.")
        return {**out, "intent": "find", "target": place, "prefer": lock.prefer, "method": "place",
                "say": f"Looking for {target_phrase(place)}. {how}"}

    prefer = "nearest" if NEAREST_WORDS.search(clean) else None
    query = NEAREST_WORDS.sub(" ", clean) if prefer else text  # the rules reject "nearest"; LiDAR handles it
    r = interpreter.interpret(query)
    out["method"] = r.get("method")
    status, intent = r.get("status"), r.get("intent")

    def start_find(target, side):
        spec = resolve_target(target, classes)
        if spec is None:
            return {**out, "intent": "unsupported", "say": f"I can't detect {with_article(target)} yet."}
        side = "center" if side in ("centre", "middle") else side
        lock.find(spec, side=side, prefer=prefer)
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
        if intent == "read_text":
            if not session.get("ocr"):
                return {**out, "intent": "unsupported", "say": "Reading signs isn't available on the server."}
            session["read_pending"] = True   # read while the user turns (sign_reader.py); the answer comes as "text_result"
            return {**out, "intent": "read", "say": ""}
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
    classes = _base_classes(args)
    extra = [c for c in EXTRA_CLASSES if c not in classes]
    if extra:
        print(f"Adding sign classes: {', '.join(extra)}")
    return classes + extra


def _base_classes(args):
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


def build_app(detector, interpreter=None, webui_dir=None, llm=None, vlm=None):
    from concurrent.futures import ThreadPoolExecutor

    from aiohttp import WSMsgType, web

    state = {"client": None}
    gpu = asyncio.Lock()
    interpreter = interpreter or VoiceInterpreter(detector.classes, webui_dir)
    voice_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice")
    way_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="way")   # way_around.py LLM calls

    async def warm_voice(_app):
        loop = asyncio.get_running_loop()
        loop.run_in_executor(voice_pool, interpreter.warmup)
        if llm is not None:
            loop.run_in_executor(voice_pool, llm.warmup)
        if vlm is not None:
            loop.run_in_executor(None, vlm.check)

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

    async def handle_blocked(ws, m, session):
        """Phone: something blocks the way to the locked target. In a room: "Processing, wait."
        once, then the LLM's checked answer (way_around.py). Elsewhere: the phone's own line.
        Runs as its own task so frames keep flowing while the LLM thinks."""
        rid = m.get("request_id")
        try:
            now = time.monotonic()
            place = way_around.place_of(session, now)
            if place not in way_around.ROOM_PLACES:
                print(f"Way: place={place}, not a room -> phone's default line")
                await safe_send(ws, {"type": "way", "request_id": rid, "status": "default", "place": place})
                return
            data = way_around.build_input(m, session.get("recent_objects"), now)
            print(f"Way input: {json.dumps(data, separators=(',', ':'))}")
            if llm is not None:
                await safe_send(ws, {"type": "way", "request_id": rid, "status": "processing",
                                     "say": way_around.PROCESSING_SAY})
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(way_pool, way_around.decide, llm, data)
            await safe_send(ws, {"type": "way", "request_id": rid, "status": "answer", **answer})
        except Exception as e:  # noqa: BLE001
            print(f"Way error: {e}")
            await safe_send(ws, {"type": "way", "request_id": rid, "status": "default", "place": "error"})

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=8_000_000, heartbeat=10)
        await ws.prepare(request)
        peer = request.remote
        if state["client"] is not None and not state["client"].closed:
            # A new phone connection replaces the old one (e.g. app restarted).
            await state["client"].close()
        state["client"] = ws
        lock = TargetLock()
        session = {"candidates": [], "lock": lock, "tracking": detector.new_session(), "scene": SceneMemory(),
                   "ocr_cache": {}, "read_pending": False, "last_ocr": 0.0,
                   "ocr": getattr(detector, "ocr", None) is not None,
                   "place": place_vlm.PlaceContext(vlm)}   # where the user is (place_vlm.py)
        search = PlaceSearch(detector, lock, session, detector.classes, llm)   # place_search.py
        session["search"] = search
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
                                voice_pool, voice_turn, text, interpreter, lock, session, detector.classes, llm)
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
                            session["read_pending"] = False
                            session["reading"] = None
                            for g in search.on_target(action, m.get("reason"), time.monotonic()):
                                await safe_send(ws, g)
                        elif action == "find":
                            search.on_target(action, None, time.monotonic())
                            spec = resolve_target(m.get("label"), detector.classes)
                            if spec:
                                lock.find(spec, side=m.get("side"), exclude=m.get("exclude") or (),
                                          prefer=m.get("prefer"))
                        elif action == "another":
                            lock.another()
                    elif kind == "blocked":
                        asyncio.create_task(handle_blocked(ws, m, session))
                elif msg.type == WSMsgType.BINARY:
                    meta, pending_meta = pending_meta, None
                    if meta is None:
                        continue
                    fid = meta.get("id")
                    try:
                        async with gpu:
                            objects, tracking, read, det_ms, srv_ms = await loop.run_in_executor(
                                None, process_frame, detector, session, meta, msg.data, confidence)
                    except Exception as e:  # noqa: BLE001
                        print(f"Frame {fid} failed: {e}")
                        await safe_send(ws, {"type": "error", "id": fid, "message": str(e)[:200]})
                        continue
                    now = time.monotonic()
                    target = lock.update(objects, now, tracking, frame=session.get("last_frame"))
                    session["recent_objects"] = (now, objects)   # names for way_around.py
                    if target is not None and target.get("state") != session.get("last_state"):
                        d = target.get("distance_m")
                        print(f"Target: {target['label']} -> {target['state']}"
                              + (f" (#{target.get('track_id')} {target.get('matched')}, "
                                 f"{'?' if d is None else f'{d:.1f} m'})" if target["state"] in ("acquired", "tracking") else ""))
                        session["last_state"] = target.get("state")
                    await safe_send(ws, {"type": "result", "id": fid, "objects": objects, "target": target,
                                         "place": session["place"].current(now)["place"],
                                         "detector_ms": round(det_ms, 1), "server_ms": round(srv_ms, 1)})
                    if search.planner is not None or lock.spec:   # place search, or the re-find watch
                        try:
                            guides = await loop.run_in_executor(None, search.on_frame, target, now)
                        except Exception as e:  # noqa: BLE001
                            print(f"Search planner error: {e}")
                            guides = []
                        for g in guides:
                            if g.get("say"):
                                print(f"Guide [{g['stage']}]: {g['say']}")
                            await safe_send(ws, g)
                    if read is not None:
                        print(f"Read: {read['say']}")
                        await safe_send(ws, {"type": "text_result", "say": read["say"], "groups": read["groups"]})
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            search.close()
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
    p.add_argument("--no-ocr", action="store_true", help="don't load the sign reader (PP-OCRv5)")
    p.add_argument("--llm-model", default="qwen3.5:4b", help="Ollama model for complex voice commands")
    p.add_argument("--llm-url", default="http://127.0.0.1:11434", help="Ollama address")
    p.add_argument("--no-llm", action="store_true", help="rules only, no LLM")
    p.add_argument("--vlm-url", default="http://127.0.0.1:8080/v1",
                   help="OpenAI-style VLM server for 'where am I' (zrt serve ... --label vlm)")
    p.add_argument("--vlm-model", default="vlm", help="served model name (see /v1/models)")
    p.add_argument("--no-vlm", action="store_true", help="no camera model: the place is always 'unknown'")
    p.add_argument("--ocr-confidence", type=float, default=0.6, help="minimum OCR text score (default 0.6)")
    p.add_argument("--cert", help="TLS certificate (then use wss:// in the app)")
    p.add_argument("--key", help="TLS private key")
    args = p.parse_args()

    from aiohttp import web
    print(f"LiDAR Alert fusion server, version {VERSION}")
    classes = load_classes(args)
    print(f"Loading {args.model} ...")
    detector = Detector(args.model, classes, args.imgsz, args.tracker, args.device, args.webui_dir,
                        ocr=not args.no_ocr, ocr_confidence=args.ocr_confidence)
    ssl_ctx = None
    if args.cert and args.key:
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(args.cert, args.key)
    scheme = "wss" if ssl_ctx else "ws"
    print(f"Ready. App server address: {scheme}://<this machine's IP>:{args.port}/ws")
    if args.no_llm:
        llm = None
    elif LLMIntent is None:
        print("LLM: intent_llm.py not found next to fusion_server.py; complex commands use the rules only")
        llm = None
    else:
        llm = LLMIntent(args.llm_url, args.llm_model, classes)
    vlm = None if args.no_vlm else place_vlm.VLMClient(args.vlm_url, args.vlm_model)
    web.run_app(build_app(detector, webui_dir=args.webui_dir, llm=llm, vlm=vlm), host=args.host, port=args.port, ssl_context=ssl_ctx, print=None)


if __name__ == "__main__":
    sys.exit(main())
