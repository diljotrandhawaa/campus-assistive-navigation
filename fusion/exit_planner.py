""""Help me find an exit": decide room vs hallway, then set the target.

Room/restroom -> any door (one that says EXIT wins). Hallway -> exit sign first, then the
door by it. Rules decide the place; the LLM (intent_llm.plan_place) only when they can't tell.
"""
import re
import time

from scene_memory import SceneMemory, classify_place

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


EXIT_SAY = {
    "room": "You seem to be in a room: {why}. I'll find the door.",
    "restroom": "You seem to be in a restroom: {why}. I'll find the door.",
    "hallway": "You seem to be in a hallway: {why}. I'll look for an exit sign, then the door by it.",
    "unknown": "I'm not sure where you are yet. Turn slowly. I'll look for an exit sign or a door.",
    "asked_room": "I'll find the door of this room.",
    "asked_building": "Looking for the building exit. I'll find an exit sign, then the door by it.",
}


def plan_exit(text, clean, lock, session, classes, llm=None):
    """"Help me find an exit": decide room vs hallway (rules; LLM only if the rules can't tell)."""
    scene = session.get("scene")
    summary = scene.summary(time.monotonic()) if scene else {"frames": 0, "counts": {}, "texts": [],
                                                            "hallway_views": 0, "views": 0}
    source, say_key = "rule", None
    if ROOM_EXIT.search(clean):
        place, why, say_key = "room", "", "asked_room"
    elif BUILDING_EXIT.search(clean):
        place, why, say_key = "hallway", "", "asked_building"
    else:
        place, why = classify_place(summary)
        if place == "unknown" and llm is not None and hasattr(llm, "plan_place") and summary["frames"]:
            try:
                d = llm.plan_place(text, SceneMemory.describe(summary))
                if d and d.get("place") in ("room", "restroom", "hallway"):
                    place, why, source = d["place"], d.get("reason") or "from what I've seen", "llm"
            except Exception as e:  # noqa: BLE001
                print(f"LLM: exit planning failed ({e}); using the rules")
    spec = exit_spec(place, classes)
    lock.find(spec)
    session["candidates"] = []
    print(f"Exit plan ({source}): place={place} target={spec['label']} | {SceneMemory.describe(summary)}")
    return {"type": "voice_intent", "transcript": text, "intent": "find", "target": spec["label"],
            "exclude": [], "side": None, "prefer": lock.prefer, "candidates": [], "method": f"exit-{source}",
            "place": place, "say": EXIT_SAY[say_key or place].format(why=why)}
