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

try:  # the LLM part is optional: without intent_llm.py the server runs rules-only
    from intent_llm import LLMIntent, clean_llm
except ImportError:
    LLMIntent = clean_llm = None

PROTOCOL = "fusion-1"
VERSION = "2026-09-25 voice+find+cameratracker+bearing+signs-ocr+llm(intent_llm.py)"

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

    def bearing(self, box):
        """Horizontal direction of the box centre relative to the walking direction, in degrees
        (+ = right). Uses only the camera pose and lens, so it works at any distance, even
        beyond LiDAR range."""
        u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        px, py = v * self.w, (1 - u) * self.h          # portrait (u, v) -> sensor pixel
        ray = self.T[:3, :3] @ np.array([(px - self.cx) / self.fx, -(py - self.cy) / self.fy, -1.0])
        flat = np.array([ray[0], ray[2]])
        if np.linalg.norm(flat) < 1e-6:
            return None
        return float(np.degrees(np.arctan2(self.right @ flat, self.forward @ flat)))

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


def text_words(text):
    t = (text or "").lower().replace("’", "").replace("'", "")
    return re.findall(r"[a-z0-9]+", t)


def text_matches(text, keywords):
    """Whole-word match, so "women" does not match "men" and "204" does not match "2045"."""
    tokens = text_words(text)
    if not tokens:
        return False
    joined = f" {' '.join(tokens)} "
    for k in keywords:
        kt = " ".join(text_words(k))
        if kt and (f" {kt} " in joined):
            return True
    return False


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


def spoken_feet(meters):
    feet = meters * 3.28084
    if feet < 1:
        return "under 1 foot"
    n = int(round(feet))
    return "1 foot" if n == 1 else f"{n} feet"


def direction_words(bearing):
    if bearing is None:
        return "ahead"
    side = "left" if bearing < 0 else "right"
    a = abs(bearing)
    return ("straight ahead" if a < 8 else f"slightly {side}" if a < 25
            else f"to your {side}" if a < 60 else f"far {side}")


def group_text_items(items):
    """Joins OCR lines that belong to the same sign (close together in the image)."""
    groups = []
    for it in sorted(items, key=lambda i: (i["box"][1], i["box"][0])):
        cx = (it["box"][0] + it["box"][2]) / 2
        for g in groups:
            gx = (g["box"][0] + g["box"][2]) / 2
            if abs(cx - gx) < 0.2 and it["box"][1] - g["box"][3] < 0.08:
                g["texts"].append(it["text"])
                g["box"] = [min(g["box"][0], it["box"][0]), min(g["box"][1], it["box"][1]),
                            max(g["box"][2], it["box"][2]), max(g["box"][3], it["box"][3])]
                g["near"] = g["near"] or (it.get("nearby_object") or {}).get("label")
                break
        else:
            groups.append({"texts": [it["text"]], "box": list(it["box"]),
                           "near": (it.get("nearby_object") or {}).get("label")})
    return groups


# "Lab 3B", "Room 204", "RM 12", or a bare 2-4 digit number like "204"
LABELED_NUMBER = re.compile(r"\b(room|rm|lab|laboratory|office|classroom|suite)\.?\s*#?\s*([a-z]?\d{1,4}[a-z]?)\b", re.I)
BARE_NUMBER = re.compile(r"\b([a-z]?\d{2,4}[a-z]?)\b", re.I)


def where_text(g):
    where = direction_words(g.get("bearing_deg"))
    return f"{spoken_feet(g['distance_m'])}, {where}" if g.get("distance_m") is not None else where


def answer_which(groups, room_kind=None):
    """"What room / lab is this?" -> the number on the nearest sign."""
    for g in groups:
        text = " ".join(g["texts"])
        m = LABELED_NUMBER.search(text)
        if m:
            kind = {"rm": "room", "laboratory": "lab"}.get(m.group(1).lower(), m.group(1).lower())
            number = m.group(2)
        else:
            m = BARE_NUMBER.search(text)
            if not m:
                continue
            kind, number = room_kind or "room", m.group(1)
        return f"This is {kind} {number.upper()}. The sign says: {text}. It's {where_text(g)}."
    if groups:
        return f"I can't find a number. The nearest sign says: {' '.join(groups[0]['texts'])}."
    return "I don't see a sign. Point the phone at the sign next to the door and ask again."


def answer_verify(groups, keywords, label=None):
    """"Is this the chemistry lab?" -> yes/no from the sign text."""
    if not groups:
        return "I can't see a sign to check. Point the phone at it and ask again."
    for g in groups:
        text = " ".join(g["texts"])
        if text_matches(text, keywords):
            return f"Yes. The sign says: {text}. It's {where_text(g)}."
    return f"I don't think so. The nearest sign says: {' '.join(groups[0]['texts'])}."


def describe_read(groups):
    if not groups:
        return "I don't see any readable text. Point the phone at the sign and ask again."
    parts = []
    for g in groups[:3]:
        where = direction_words(g.get("bearing_deg"))
        if g.get("distance_m") is not None:
            where = f"{spoken_feet(g['distance_m'])}, {where}"
        thing = f"On the {g['near']}" if g.get("near") else "Sign"
        parts.append(f"{thing}, {where}: {'. '.join(g['texts'])}.")
    return " ".join(parts)


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
        self.classes = classes
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

    def new_session(self):
        """Tracking state for one phone connection."""
        if self.camera_tracker_cls:
            return {"kind": "camera", "tracker": self.camera_tracker_cls()}
        self.model.predictor = None  # fresh built-in tracker state
        return {"kind": "builtin", "target": None}

    def __call__(self, bgr, confidence, session, target):
        """Returns (detections, tracking_info). `target` is a set of class names (or None);
        only those detections get track ids."""
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
    if getattr(detector, "ocr", None) is not None and spec and spec["keywords"]:
        read_target_text(detector, sess, bgr, objects, tracking, now)

    read = None
    if sess.get("read_pending"):
        sess["read_pending"] = False
        if getattr(detector, "ocr", None) is None:
            read = {"say": "Reading signs isn't available on the server.", "groups": []}
        else:
            groups = group_text_items(detector.read_full(bgr, objects))
            for g in groups:
                d, _, _ = frame.measure(g["box"])
                g["distance_m"] = None if d is None else round(d, 2)
                b = frame.bearing(g["box"])
                g["bearing_deg"] = None if b is None else round(b, 1)
            groups.sort(key=lambda g: g["distance_m"] if g["distance_m"] is not None else 99)
            mode = sess.get("read_mode") or {}
            if mode.get("kind") == "verify":
                say = answer_verify(groups, mode.get("keywords", []), mode.get("label"))
            elif mode.get("kind") == "which":
                say = answer_which(groups, mode.get("room_kind"))
            else:
                say = describe_read(groups)
            read = {"say": say, "groups": groups}
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
            print(f"OCR: #{o['track_id']} {o['label']} reads {text!r}")


# ----------------------------------------------------------------------------- find mode
class TargetLock:
    """Find mode: lock onto ONE tracked candidate of the target spec and report it every frame.

    Same idea as yolo_live/voice_direction.TargetController (lock a BoT-SORT id, tolerate
    short dropouts), but the report is metric (distance + sideways offset from LiDAR).
    Candidates: objects of the spec's classes. With keywords (places), a candidate whose
    OCR text matches is preferred; "specific" classes (e.g. "restroom sign") count on their
    own; generic ones (sign, door) only count when their text matches.
    """

    LOST_GRACE = 3.0  # seconds without an id update before searching again (built-in tracker)

    def __init__(self):
        self.cancel()

    def cancel(self):
        self.spec = None
        self.label = None
        self.side = None          # "left" / "center" / "right" / None
        self.exclude = set()      # labels to ignore (filled by the future LLM parser)
        self.prefer = "nearest"
        self.skip_ids = set()     # track ids rejected with "another one"
        self.track_id = None
        self.last_seen = None
        self.ever_seen = False
        self.generation = None
        self.smooth = None        # (distance, lateral) of the locked object, smoothed
        self.smooth_bearing = None

    def find(self, spec, side=None, exclude=(), prefer=None):
        self.cancel()
        self.spec = spec
        self.label = spec["label"]
        self.side = side
        self.exclude = set(exclude or ())
        self.prefer = prefer or "nearest"

    def another(self):
        if self.track_id is not None:
            self.skip_ids.add(self.track_id)
        self._unlock()

    def set_side(self, side):
        self.side = side
        self._unlock()

    def _unlock(self):
        self.track_id = None
        self.last_seen = None
        self.smooth = None
        self.smooth_bearing = None

    def _bearing(self, obj):
        b = obj.get("bearing_deg")
        if b is None:
            return None
        if self.smooth_bearing is not None and abs(self.smooth_bearing - b) < 20:
            b = 0.5 * self.smooth_bearing + 0.5 * b
        self.smooth_bearing = b
        return round(b, 1)

    def _smoothed(self, obj):
        d, l = obj["distance_m"], obj["lateral_m"]
        if d is None or l is None:
            return d, l
        if self.smooth and abs(self.smooth[0] - d) < 0.8:
            d = 0.5 * self.smooth[0] + 0.5 * d
            l = 0.5 * self.smooth[1] + 0.5 * l
        self.smooth = (d, l)
        return round(d, 3), round(l, 3)

    @staticmethod
    def side_of(obj):
        s = obj.get("side")
        if s:
            return "center" if s == "ahead" else s
        u = (obj["box"][0] + obj["box"][2]) / 2
        return "left" if u < 1 / 3 else "right" if u > 2 / 3 else "center"

    def _score(self, o):
        """2 = confirmed by sign text, 1 = counts by class alone, None = not a candidate."""
        if o["label"] not in self.spec["classes"] or o["label"] in self.exclude:
            return None
        if o.get("track_id") is None or o["track_id"] in self.skip_ids:
            return None
        if self.spec["keywords"] and text_matches(o.get("text"), self.spec["keywords"]):
            return 2
        if o["label"] in self.spec["specific"]:
            return 1
        return None

    def _near_ok(self, o, objects):
        """"the sofa near the window": another object of a `near` class within `within_m`,
        measured on the floor plane from the LiDAR positions (ahead, sideways)."""
        near = self.spec.get("near")
        if not near:
            return True
        if o.get("distance_m") is None or o.get("lateral_m") is None:
            return False
        for other in objects:
            if (other is not o and other["label"] in near["classes"]
                    and other.get("distance_m") is not None and other.get("lateral_m") is not None
                    and np.hypot(other["distance_m"] - o["distance_m"],
                                 other["lateral_m"] - o["lateral_m"]) <= near["within_m"]):
                return True
        return False

    def _detail(self, o):
        """What was actually found, when it isn't simply the target class ("restroom sign",
        'door, sign reads "Women"')."""
        if o["label"] == self.label and not o.get("text"):
            return None
        if o.get("text"):
            return f'{o["label"]} that says "{o["text"][:40]}"'
        return o["label"]

    def _report(self, state, o):
        d, l = self._smoothed(o)
        return {"label": self.label, "side_filter": self.side, "state": state, "track_id": o["track_id"],
                "matched": o["label"], "text": o.get("text"), "detail": self._detail(o),
                "distance_m": d, "lateral_m": l, "side": side_text(l), "bearing_deg": self._bearing(o)}

    def update(self, objects, now, tracking=None):
        """tracking: CameraTracker info {"generation","alive_ids","retention_seconds"} or None."""
        if not self.spec:
            return None
        base = {"label": self.label, "side_filter": self.side}
        if tracking is not None:
            if tracking.get("generation") != self.generation:
                # Tracker was rebuilt (new target, new image size, or long gap): old ids are meaningless.
                self.generation = tracking.get("generation")
                self.skip_ids.clear()
                self._unlock()

        if self.track_id is not None:
            for o in objects:
                if o.get("track_id") == self.track_id:
                    self.last_seen = now
                    o["is_target"] = True
                    return self._report("tracking", o)
            if tracking is not None:
                # CameraTracker keeps an unseen id alive (ReID can re-match it) for its retention time.
                still_alive = (self.track_id in tracking.get("alive_ids", [])
                               and now - self.last_seen <= tracking.get("retention_seconds", 10.0))
            else:
                still_alive = now - self.last_seen <= self.LOST_GRACE
            if still_alive:
                return {**base, "state": "lost_briefly", "track_id": self.track_id}
            self._unlock()  # gone: fall through and search again

        scored = [(self._score(o), o) for o in objects]
        eligible = [(sc, o) for sc, o in scored
                    if sc is not None and (self.side is None or self.side_of(o) == self.side)
                    and self._near_ok(o, objects)]
        if not eligible:
            return {**base, "state": "searching", "seen_before": self.ever_seen}
        best = max(sc for sc, _ in eligible)  # text-confirmed candidates win
        pool = [o for sc, o in eligible if sc == best]
        if self.prefer == "nearest":
            pick = min(pool, key=lambda o: (o["distance_m"] is None, o["distance_m"] or 0, -o["confidence"]))
        else:
            pick = max(pool, key=lambda o: o["confidence"])
        self.track_id, self.last_seen, self.ever_seen = pick["track_id"], now, True
        self.smooth = None
        self.smooth_bearing = None
        pick["is_target"] = True
        return self._report("acquired", pick)

    def status_text(self):
        if not self.label:
            return "Not searching for anything."
        where = f" on your {self.side}" if self.side in ("left", "right") else ""
        return f"Looking for {target_phrase(self.label)}{where}."


def target_phrase(label):
    return label if label.startswith("room ") else with_article(label)


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


def rules_reply(text, interpreter, lock, session, classes):
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
        session["read_pending"] = True   # the next frame is read; the answer comes as "text_result"
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
            session["read_pending"] = True   # the next frame is read; the answer comes as "text_result"
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


def build_app(detector, interpreter=None, webui_dir=None, llm=None):
    from concurrent.futures import ThreadPoolExecutor

    from aiohttp import WSMsgType, web

    state = {"client": None}
    gpu = asyncio.Lock()
    interpreter = interpreter or VoiceInterpreter(detector.classes, webui_dir)
    voice_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice")

    async def warm_voice(_app):
        loop = asyncio.get_running_loop()
        loop.run_in_executor(voice_pool, interpreter.warmup)
        if llm is not None:
            loop.run_in_executor(voice_pool, llm.warmup)

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
        lock = TargetLock()
        session = {"candidates": [], "lock": lock, "tracking": detector.new_session(),
                   "ocr_cache": {}, "read_pending": False, "last_ocr": 0.0,
                   "ocr": getattr(detector, "ocr", None) is not None}
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
                                voice_pool, voice_reply, text, interpreter, lock, session, detector.classes, llm)
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
                        elif action == "find":
                            spec = resolve_target(m.get("label"), detector.classes)
                            if spec:
                                lock.find(spec, side=m.get("side"), exclude=m.get("exclude") or (),
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
                            objects, tracking, read, det_ms, srv_ms = await loop.run_in_executor(
                                None, process_frame, detector, session, meta, msg.data, confidence)
                    except Exception as e:  # noqa: BLE001
                        print(f"Frame {fid} failed: {e}")
                        await safe_send(ws, {"type": "error", "id": fid, "message": str(e)[:200]})
                        continue
                    target = lock.update(objects, time.monotonic(), tracking)
                    await safe_send(ws, {"type": "result", "id": fid, "objects": objects, "target": target,
                                         "detector_ms": round(det_ms, 1), "server_ms": round(srv_ms, 1)})
                    if read is not None:
                        print(f"Read: {read['say']}")
                        await safe_send(ws, {"type": "text_result", "say": read["say"], "groups": read["groups"]})
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
    p.add_argument("--no-ocr", action="store_true", help="don't load the sign reader (PP-OCRv5)")
    p.add_argument("--llm-model", default="qwen3.5:4b", help="Ollama model for complex voice commands")
    p.add_argument("--llm-url", default="http://127.0.0.1:11434", help="Ollama address")
    p.add_argument("--no-llm", action="store_true", help="rules only, no LLM")
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
    web.run_app(build_app(detector, webui_dir=args.webui_dir, llm=llm), host=args.host, port=args.port, ssl_context=ssl_ctx, print=None)


if __name__ == "__main__":
    sys.exit(main())
