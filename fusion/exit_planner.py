""""Help me find an exit": decide room vs hallway, then set the target.

Room/restroom -> any door (one that says EXIT wins). Hallway -> exit sign first, then the
door by it. The place comes from the camera model (place_vlm.py); the user's own words win.
"""
import re
import time

from place_vlm import scene_hints
from scene_memory import SceneMemory

PLACE_WAIT_S = 4.0   # wait this long for a fresh VLM answer when the last one is old

EXIT_REQUEST = re.compile(r"\b(?:exit|way out|get out|leave (?:this|the) (?:room|building|place)|go outside)\b")


ROOM_EXIT = re.compile(r"\b(?:this|the) (?:room|classroom|lab|office)\b")


BUILDING_EXIT = re.compile(r"\b(?:building|outside|emergency|fire exit|main exit|front door)\b")


def exit_spec(place, classes):
    classes = set(classes)
    if place in ("room", "restroom"):
        # Leave this room: any door; a door that says EXIT wins.
        return {"label": "door", "classes": {"door"} & classes, "keywords": ["exit"],
                "specific": {"door"} & classes}
    spec = {"label": "exit", "classes": {"exit sign", "sign", "door"} & classes, "keywords": ["exit"],
            "specific": {"exit sign"} & classes,
            "goal": {"classes": {"door"} & classes, "near_clue_m": 2.0}}   # sign first, then the door by it
    if place == "unknown":
        spec["fallback"] = {"door"} & classes                              # any door is better than nothing
    return spec


EXIT_SAY = {   # short on purpose: where the user is, and what happens next (no list of what was seen)
    "room": "You're in a room. I'll find the door.",
    "restroom": "You're in a restroom. I'll find the door.",
    "hallway": "You're in a hallway. I'll look for an exit sign, then the door by it.",
    "unknown": "I'm not sure where you are yet. Turn slowly. I'll look for an exit sign or a door.",
    "asked_room": "I'll find the door of this room.",
    "asked_building": "Looking for the building exit. I'll find an exit sign, then the door by it.",
}


def plan_exit(text, clean, lock, session, classes, llm=None):  # noqa: ARG001  (llm: kept for callers)
    """"Help me find an exit": room vs hallway from the VLM (place_vlm.py); explicit words win."""
    scene = session.get("scene")
    now = time.monotonic()
    summary = scene.summary(now) if scene else {"frames": 0, "counts": {}, "texts": [],
                                                 "hallway_views": 0, "views": 0}
    source, say_key = "vlm", None
    if ROOM_EXIT.search(clean):
        place, why, say_key, source = "room", "", "asked_room", "asked"
    elif BUILDING_EXIT.search(clean):
        place, why, say_key, source = "hallway", "", "asked_building", "asked"
    else:
        ctx = session.get("place")
        p = ctx.get(now, lambda: scene_hints(scene, now), wait=PLACE_WAIT_S) if ctx is not None else None
        place = p["planner_place"] if p else "unknown"
        why = ""
        if p:
            source = f"vlm {p['place']}"
    spec = exit_spec(place, classes)
    lock.find(spec)
    session["candidates"] = []
    print(f"Exit plan ({source}): place={place} target={spec['label']} | {SceneMemory.describe(summary)}")
    return {"type": "voice_intent", "transcript": text, "intent": "find", "target": spec["label"],
            "exclude": [], "side": None, "prefer": lock.prefer, "candidates": [], "method": f"exit-{source}",
            "place": place, "say": EXIT_SAY[say_key or place].format(why=why)}
