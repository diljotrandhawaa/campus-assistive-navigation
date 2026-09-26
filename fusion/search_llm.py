"""The LLM question of a place search (local Ollama model via intent_llm.LLMIntent).

where_next(): after searching this area without success -> what to suggest next
(the planner always ASKS the user before leading them to the door).
Returns a validated dict or None; the planner falls back to its rules on None.
Where the user is comes from place_vlm.py (the camera model), not from here.
"""
import json
import re
import time

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
