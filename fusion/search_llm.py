"""The two one-time LLM questions of a place search (local Ollama model via intent_llm.LLMIntent).

1. plan_target(): at the request, only when the rules need help -> where the user probably is,
   and extra sign phrases for YOLOE to look for.
2. where_next(): after searching this area without success -> what to suggest next
   (the planner always ASKS the user before leading them to the door).

Both return a validated dict or None; the planner falls back to its rules on None.
"""
import json
import re
import time

PLAN_SYSTEM = """A blind user is looking for a place (for example a restroom or a room number) with a camera.
From what the camera saw recently, answer:
place: room (classroom, office, lab, meeting room), restroom, hallway (corridor, lobby, stairwell) or unknown.
prompts: up to 5 short phrases (2-4 words) an object detector could use to spot the place's SIGN or entrance,
e.g. for a restroom: "toilet sign", "men's room sign". Only use what is listed; if unsure, place is unknown."""

PLAN_SCHEMA = {"type": "object", "properties": {
    "place": {"type": "string", "enum": ["room", "restroom", "hallway", "unknown"]},
    "prompts": {"type": "array", "items": {"type": "string"}}},
    "required": ["place", "prompts"]}

NEXT_SYSTEM = """A blind user asked for a place. The camera searched all around the current area and did not find it.
Suggest the next step. next: leave_room (it is probably outside this room, e.g. restrooms are off hallways),
keep_looking (it could still be here) or give_up. say: one short sentence explaining why, spoken to the user,
without asking a question (the app will ask whether to go to the door)."""

NEXT_SCHEMA = {"type": "object", "properties": {
    "next": {"type": "string", "enum": ["leave_room", "keep_looking", "give_up"]},
    "say": {"type": "string"}},
    "required": ["next", "say"]}


def _clean_sentence(s, limit=140):
    s = re.sub(r"[^\w\s,.'-]", "", str(s or "")).strip()
    return s[:limit]


def plan_target(llm, question, label, scene_text):
    t0 = time.perf_counter()
    r = llm._post([{"role": "system", "content": PLAN_SYSTEM},
                   {"role": "user", "content": f'Question: "{question}"\nLooking for: {label}\n{scene_text}'}],
                  num_predict=80, schema=PLAN_SCHEMA)
    d = json.loads(r["message"]["content"])
    print(f"LLM search plan: {json.dumps(d)} ({time.perf_counter() - t0:.2f} s)")
    if not isinstance(d, dict) or d.get("place") not in ("room", "restroom", "hallway", "unknown"):
        return None
    prompts = [p for p in (d.get("prompts") or []) if isinstance(p, str)][:5]
    return {"place": d["place"], "prompts": prompts}


def where_next(llm, label, scene_text, searched_text):
    t0 = time.perf_counter()
    r = llm._post([{"role": "system", "content": NEXT_SYSTEM},
                   {"role": "user", "content": f"Looking for: {label}\n{searched_text}\n{scene_text}"}],
                  num_predict=70, schema=NEXT_SCHEMA)
    d = json.loads(r["message"]["content"])
    print(f"LLM where next: {json.dumps(d)} ({time.perf_counter() - t0:.2f} s)")
    if not isinstance(d, dict) or d.get("next") not in ("leave_room", "keep_looking", "give_up"):
        return None
    return {"next": d["next"], "say": _clean_sentence(d.get("say"))}
