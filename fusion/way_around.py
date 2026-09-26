"""Way around an obstacle on the way to a locked target, in a room.

The phone notices the blockage (its 3 s LiDAR free-space memory, FreeSpace.swift) and sends
what it measured. Here: in a room, the local LLM decides which way to go from those
measurements; the code checks the answer against the same measurements and says it in the
shortest words. Anywhere else (hallway / unknown) the phone keeps its own line.
Where the user is: place_vlm.py (the camera model).

    phone  -> {"type":"blocked","request_id":N,"obstacle":"table"|null,
               "target":{"label","distance_m","bearing_deg","width_m"},
               "layout":{"sweep":[{"deg":-90,"free_m":2.5|null},...],
                         "obstacle":{"near_m","far_m","left_m","right_m","left_seen","right_seen"}|null,
                         "gap_left_m":1.2|null,"gap_right_m":null}}
    server -> {"type":"way","request_id":N,"status":"default","place":"hallway"}      (not a room)
    server -> {"type":"way","request_id":N,"status":"processing","say":"Processing, wait."}   (once)
    server -> {"type":"way","request_id":N,"status":"answer","say":"Table ahead, gap on the right.",
               "action":"around|sidestep|look|blocked","side":"left|right"|null,"steps":N|null,
               "method":"llm|rule ...","ms":850}

Layout frame (from the phone): meters from the user along the DIRECTION OF THE TARGET.
"ahead" = toward the target; lateral / side values: negative = left, positive = right.

Spoken answers (all built here, never free text from the model):
    around    "Table ahead, gap on the right."                 (obstacle 1 m or more away)
    sidestep  "Chair ahead, gap on the left."                  (obstacle closer than 1 m; same words)
    look      "Table ahead. No gap found yet."                 (no gap seen yet)
    blocked   "Table is blocking the way. No gap found."       (both sides seen and too narrow)
No turn instructions for obstacles ("turn left", "walk 3 steps", ...): the sentence only says where
the gap is; the phone's MOVE vibration comes on when the user faces the free way (Detour follows s).
The LLM never decides STOP; the phone's LiDAR does. On timeout / bad output: the rules below.
"""
import json
import math
import time


PROCESSING_SAY = "Processing, wait."
ROOM_PLACES = {"room"}      # where the LLM is asked; elsewhere the phone keeps its own line
CLOSE_M = 1.0               # obstacle nearer than this: turn, side-step, turn back
MIN_GAP_M = 0.8             # a walking person needs this much free width
STEP_M = 0.7                # one walking step
BODY_CLEAR_M = 0.45         # half a person plus a margin, past the obstacle's edge
MAX_STEPS = 8
OBJECT_MAX_AGE = 1.0        # s: detections older than this are not used for names
SIDES = ("left", "right")
# Field test 2026-09-26: the spoken side came out opposite (gap on the left -> "right").
# True = swap left <-> right in the final answer (speech, the side the phone steers to, steps).
# Set False to go back to the original sides.
SWAP_SIDES = True
ACTIONS = ("around", "sidestep", "look", "blocked")

WAY_SYSTEM = """You guide a blind person who is walking to a target inside a room. Something blocks the straight way.
From the measurements, decide how to get around it. Reply with JSON only.

Frame: meters from the user, measured along the direction of the target. "ahead" = toward the target.
Side values: negative = left, positive = right. The user is roughly facing the target.

Data:
- target: name, distance_m (null = far), bearing_deg (negative = the target is a bit to the left).
- blocking: the obstacle on the straight line: distance_m to its near side, depth_m, left_edge_m and right_edge_m
  (where its sides are), left_edge_seen / right_edge_seen (an unseen side may continue further).
- gap_left_m / gap_right_m: free width beside that side of the obstacle; null = not seen yet.
  A person needs at least 0.8 m.
- steps_left / steps_right: sideways steps needed to clear that side.
- free_m_by_direction: free walking distance per direction, in degrees from the target direction
  (negative = left); null = not seen.
- other_objects: other things nearby with ahead_m and side_m. People move: avoid passing close to them.
- close: true when the obstacle is less than 1 m away.

Keys: a = action, s = side ("left", "right" or null), o = the obstacle's name from the data (or null).
Actions (the app turns them into the shortest spoken sentence):
- around: close is false. Keep walking toward the target and pass the obstacle on side s.
  The app says: "<o> ahead, gap on the <s>."
- sidestep: close is true. Turn to side s, walk steps_<s> steps, then turn back to face the target.
  The app says: "<o> ahead, gap on the <s>."
- look: no usable gap is known. The user turns slowly toward side s to check. Choose a side whose gap is
  null (not seen), preferring the side with fewer steps or the side the target is on.
  The app says: "<o> ahead. No gap found yet."
- blocked: both gaps were seen and both are under 0.8 m. s = null.
  The app says: "<o> is blocking the way. No gap found."

How to choose the side s:
1. Only a side whose gap is at least 0.8 m (never a null gap for around or sidestep).
2. Prefer the side with fewer steps (a shorter way around).
3. Prefer the side the target is on when the steps are similar (within 1).
4. Avoid a side where a person or another obstacle sits in or just past the gap.
5. A known good gap is better than looking at an unseen side.

Examples (short inputs):
{"target":{"bearing_deg":5},"blocking":{"name":"table","distance_m":1.8},"gap_left_m":0.4,"gap_right_m":1.3,"steps_left":2,"steps_right":3,"close":false}
-> {"a":"around","s":"right","o":"table"}
{"target":{"bearing_deg":-20},"blocking":{"name":"chair","distance_m":0.7},"gap_left_m":1.5,"gap_right_m":1.5,"steps_left":2,"steps_right":2,"close":true}
-> {"a":"sidestep","s":"left","o":"chair"}
{"target":{"bearing_deg":0},"blocking":{"name":"table","distance_m":1.4,"left_edge_seen":false,"right_edge_seen":true},"gap_left_m":null,"gap_right_m":0.3,"steps_left":3,"steps_right":2,"close":false}
-> {"a":"look","s":"left","o":"table"}
{"target":{"bearing_deg":0},"blocking":{"name":"sofa","distance_m":1.2},"gap_left_m":0.5,"gap_right_m":0.2,"close":false}
-> {"a":"blocked","s":null,"o":"sofa"}
{"target":{"bearing_deg":10},"blocking":{"name":"desk","distance_m":2.0},"gap_left_m":1.1,"gap_right_m":1.2,"steps_left":2,"steps_right":2,"close":false,"other_objects":[{"name":"person","ahead_m":2.3,"side_m":1.4}]}
-> {"a":"around","s":"left","o":"desk"}"""


def _num(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _r(v, nd=1):
    return None if v is None else round(v, nd)


def direction_words(bearing):
    side = "left" if bearing < 0 else "right"
    a = abs(bearing)
    if a < 8:
        return "straight ahead"
    if a < 25:
        return f"slightly {side}"
    if a < 60:
        return f"to your {side}"
    return f"far {side}"


def place_of(session, now):
    """room / restroom / hallway / unknown, from the camera model's last answer (place_vlm.py).
    Doesn't wait: place_vlm keeps it fresh in the background every few seconds."""
    ctx = session.get("place")
    return ctx.current(now)["planner_place"] if ctx is not None else "unknown"


def _steps(move_m):
    return max(1, min(MAX_STEPS, math.ceil(max(0.0, move_m) / STEP_M)))


def build_input(msg, recent_objects, now):
    """Phone measurements + the GB10's named detections -> the data the LLM (and the rules) see."""
    t = msg.get("target") or {}
    lay = msg.get("layout") or {}
    tb = _num(t.get("bearing_deg"), 0.0)

    # Named detections from the latest frame, turned into the target frame.
    named = []
    if recent_objects and now - recent_objects[0] <= OBJECT_MAX_AGE:
        for o in recent_objects[1]:
            if o.get("is_target") or o.get("at_target"):
                continue
            d, l = _num(o.get("distance_m")), _num(o.get("lateral_m"))
            if d is None or l is None:
                continue
            r = math.hypot(d, l)
            a = math.radians(math.degrees(math.atan2(l, d)) - tb)
            ahead, side = r * math.cos(a), r * math.sin(a)
            if 0 < ahead < 4.5 and abs(side) < 3.0:
                named.append({"name": str(o.get("label", ""))[:30], "ahead_m": round(ahead, 1),
                              "side_m": round(side, 1)})

    ob = lay.get("obstacle")
    blocking = None
    steps = {"left": None, "right": None}
    if isinstance(ob, dict) and _num(ob.get("near_m")) is not None:
        near, far = _num(ob.get("near_m")), _num(ob.get("far_m"), _num(ob.get("near_m")))
        left, right = _num(ob.get("left_m"), -0.3), _num(ob.get("right_m"), 0.3)
        name, best = None, None
        for n in named:   # the named thing that sits where the LiDAR obstacle is
            if near - 0.4 <= n["ahead_m"] <= far + 0.4 and left - 0.3 <= n["side_m"] <= right + 0.3:
                if best is None or n["ahead_m"] < best:
                    name, best = n["name"], n["ahead_m"]
        if name is None and msg.get("obstacle"):
            name = str(msg["obstacle"])[:30]
        blocking = {"name": name, "distance_m": _r(near), "depth_m": _r(max(0.0, far - near)),
                    "left_edge_m": _r(left), "right_edge_m": _r(right),
                    "left_edge_seen": bool(ob.get("left_seen")), "right_edge_seen": bool(ob.get("right_seen"))}
        steps = {"left": _steps(BODY_CLEAR_M - left), "right": _steps(right + BODY_CLEAR_M)}
        named = [n for n in named if n["name"] != name or not (near - 0.4 <= n["ahead_m"] <= far + 0.4)]

    sweep = {}
    for s in lay.get("sweep") or []:
        deg = _num(s.get("deg"))
        if deg is not None:
            sweep[str(int(round(deg)))] = _r(_num(s.get("free_m")))
    near = blocking["distance_m"] if blocking else _num(sweep.get("0"), 1.5)

    return {
        "target": {"name": str(t.get("label") or "target")[:30], "distance_m": _r(_num(t.get("distance_m"))),
                   "bearing_deg": round(tb), "direction": direction_words(tb)},
        "blocking": blocking,
        "gap_left_m": _r(_num(lay.get("gap_left_m"))),
        "gap_right_m": _r(_num(lay.get("gap_right_m"))),
        "steps_left": steps["left"],
        "steps_right": steps["right"],
        "close": near is not None and near < CLOSE_M,
        "free_m_by_direction": sweep,
        "other_objects": sorted(named, key=lambda n: n["ahead_m"])[:6],
    }


def _gap(data, side):
    return data.get(f"gap_{side}_m")


def _usable(data, side):
    g = _gap(data, side)
    return g is not None and g >= MIN_GAP_M


def _names(data):
    names = [(data.get("blocking") or {}).get("name")] + [n["name"] for n in data.get("other_objects", [])]
    return [n for n in dict.fromkeys(names) if n]


def _person_in_gap(data, side):
    b = data.get("blocking") or {}
    near, far = b.get("distance_m") or 0.0, (b.get("distance_m") or 0.0) + (b.get("depth_m") or 0.0)
    edge = b.get("left_edge_m" if side == "left" else "right_edge_m") or 0.0
    for n in data.get("other_objects", []):
        if n["name"] not in ("person", "people", "man", "woman", "child"):
            continue
        beyond = n["side_m"] < edge + 0.3 if side == "left" else n["side_m"] > edge - 0.3
        if beyond and near - 0.5 <= n["ahead_m"] <= far + 1.0 and abs(n["side_m"] - edge) < 1.8:
            return True
    return False


def rule_decision(data):
    """The fallback (and the yardstick for checking the LLM): fewest steps, then the target's side."""
    tb = data["target"]["bearing_deg"]
    name = (data.get("blocking") or {}).get("name")
    cands = []
    for s in SIDES:
        if _usable(data, s):
            cost = (data.get(f"steps_{s}") or 2)
            if (tb < -5 and s == "right") or (tb > 5 and s == "left"):
                cost += 1
            if _person_in_gap(data, s):
                cost += 2
            cands.append((cost, s))
    if cands:
        s = min(cands)[1]
        return {"a": "sidestep" if data["close"] else "around", "s": s, "o": name}
    unseen = [s for s in SIDES if _gap(data, s) is None]
    if unseen:
        def key(s):
            return ((data.get(f"steps_{s}") or 2), 0 if (s == "left") == (tb < 0) else 1)
        return {"a": "look", "s": min(unseen, key=key), "o": name}
    return {"a": "blocked", "s": None, "o": name}


def check(d, data):
    """The LLM's answer, or None if it doesn't fit the measurements."""
    if not isinstance(d, dict) or d.get("a") not in ACTIONS:
        return None
    a, s = d["a"], d.get("s")
    names = _names(data)
    o = d.get("o") if d.get("o") in names else (data.get("blocking") or {}).get("name")
    if a in ("around", "sidestep"):
        if s not in SIDES or not _usable(data, s):
            return None                                   # never through a gap that isn't there
        a = "sidestep" if data["close"] else "around"     # the distance decides which sentence
    elif a == "look":
        if s not in SIDES or _gap(data, s) is not None:
            return None                                   # look only toward an unseen side
        if any(_usable(data, x) for x in SIDES):
            return None                                   # a known gap beats looking
    else:  # blocked
        if any(_gap(data, x) is None or _usable(data, x) for x in SIDES):
            return None
        s = None
    return {"a": a, "s": s, "o": o}


def ask_llm(llm, data):
    names = _names(data)
    schema = {"type": "object", "properties": {
        "a": {"type": "string", "enum": list(ACTIONS)},
        "s": {"type": ["string", "null"], "enum": ["left", "right", None]},
        "o": {"type": ["string", "null"], "enum": names + [None]}},
        "required": ["a", "s", "o"]}
    r = llm._post([{"role": "system", "content": WAY_SYSTEM},
                   {"role": "user", "content": json.dumps(data, separators=(",", ":"))}],
                  num_predict=40, schema=schema)
    return json.loads(r["message"]["content"])


def phrase(dec, data):
    """The shortest spoken sentence for a checked decision."""
    raw = dec.get("o") or "something"
    name = raw[:1].upper() + raw[1:]
    a, s = dec["a"], dec.get("s")
    if a in ("around", "sidestep"):          # no "turn left / walk 3 steps": only where the gap is
        return f"{name} ahead, gap on the {s}."
    if a == "look":
        return f"{name} ahead. No gap found yet."
    return f"{name} is blocking the way. No gap found."


def decide(llm, data):
    """LLM first (if there is one), checked; rules otherwise. Returns the answer fields."""
    t0 = time.perf_counter()
    dec, method, raw = None, "rule", None
    if llm is not None:
        try:
            raw = ask_llm(llm, data)
            dec = check(raw, data)
            method = "llm" if dec else "rule (llm answer did not fit)"
        except Exception as e:  # noqa: BLE001  (timeout, Ollama down, bad JSON)
            method = f"rule (llm failed: {type(e).__name__})"
    if dec is None:
        dec = rule_decision(data)
    if SWAP_SIDES and dec.get("s") in SIDES:
        dec = {**dec, "s": "left" if dec["s"] == "right" else "right"}   # right -> left, left -> right
        method += " +swapped"
    ms = round((time.perf_counter() - t0) * 1000)
    say = phrase(dec, data)
    print(f"Way: {method} {json.dumps(raw) if raw is not None else ''} -> {dec} '{say}' ({ms} ms)")
    return {"say": say, "action": dec["a"], "side": dec.get("s"),
            "steps": None,   # no side-step instruction: the phone just steers to that side
            "method": method, "ms": ms}
