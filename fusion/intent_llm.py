#!/usr/bin/env python3
"""LLM intent parser for the LiDAR Alert fusion server.

Turns a spoken request (already converted to text on the iPhone) into a small, validated
JSON command. fusion_server.py only calls this when its rules can't handle the sentence
(negations, relations like "near the window", names, lab numbers, questions).

    parser = LLMIntent("http://127.0.0.1:11434", "qwen3.5:4b", classes)
    parser.warmup()                       # loads the model (first time ~30 s)
    d = clean_llm(parser.parse(text), classes)
    # d = {"i": "find", "t": "sofa", "x": ["chair"], "n": "window", "s": None,
    #      "p": None, "k": [], "r": None}   or None if invalid

Keys: i intent (find/read/verify/which/cancel/another/other), t target class, x excluded
classes, n class the target must be near, s side, p "nearest", k words that must be on a
sign/door, r room kind for "which room is this".

The model runs locally in Ollama (think off, temperature 0, kept loaded 24 h). Its output is
forced by a JSON schema whose enums are the server's class list, then re-checked here.
Change the model with fusion_server.py --llm-model, or disable with --no-llm.
"""
import json
import re
import time
import urllib.request


def _words(text):
    t = (text or "").lower().replace("’", "").replace("'", "")
    return re.findall(r"[a-z0-9]+", t)

# Used only when the rules above give up (negations, relations, names, lab numbers,
# questions). A small local model (Ollama) returns compact, enum-constrained JSON;
# it is validated against the class list before anything is done with it.

LLM_SYSTEM = """You turn a blind user's spoken request into JSON for a walking-assistance camera.
Keys: i=intent, t=ONE target object (or null), x=objects to exclude, n=object the target must be near,
s=side (left/right/center or null), p="nearest" or null, k=words that must be written on a sign or door
(room/lab numbers, names, department words), r=room kind for "which room" questions (room/lab/office or null).
Intents: find (go to / look for something), read (read text aloud), verify (is this X? check a sign),
which (what room/lab is this?), cancel, another (a different one of the same), other (anything else).
Use only the allowed object names. Put places that are identified by a sign (labs, offices, rooms,
departments) as t="sign" with their words in k. Lowercase k.
Examples:
"find a chair" -> {"i":"find","t":"chair","x":[],"n":null,"s":null,"p":null,"k":[],"r":null}
"not the sofa, the bench on my left" -> {"i":"find","t":"bench","x":["sofa"],"n":null,"s":"left","p":null,"k":[],"r":null}
"the closest table near the window" -> {"i":"find","t":"table","x":[],"n":"window","s":null,"p":"nearest","k":[],"r":null}
"take me to lab 3B" -> {"i":"find","t":"sign","x":[],"n":null,"s":null,"p":null,"k":["3b"],"r":null}
"I need doctor Smith's office" -> {"i":"find","t":"sign","x":[],"n":null,"s":null,"p":null,"k":["smith"],"r":null}
"is this the chemistry lab" -> {"i":"verify","t":null,"x":[],"n":null,"s":null,"p":null,"k":["chemistry"],"r":null}
"which lab is this" -> {"i":"which","t":null,"x":[],"n":null,"s":null,"p":null,"k":[],"r":"lab"}
"try a different one" -> {"i":"another","t":null,"x":[],"n":null,"s":null,"p":null,"k":[],"r":null}
"what's the weather" -> {"i":"other","t":null,"x":[],"n":null,"s":null,"p":null,"k":[],"r":null}"""


class LLMIntent:
    """Calls Ollama's /api/chat with a JSON schema; returns a dict or None (on any failure)."""

    def __init__(self, url, model, classes, timeout=3.0):
        self.url = url.rstrip("/") + "/api/chat"
        self.model = model
        self.classes = list(classes)
        self.timeout = timeout
        names = self.classes + [None]
        self.schema = {"type": "object", "properties": {
            "i": {"type": "string", "enum": ["find", "read", "verify", "which", "cancel", "another", "other"]},
            "t": {"type": ["string", "null"], "enum": names},
            "x": {"type": "array", "items": {"type": "string", "enum": self.classes}},
            "n": {"type": ["string", "null"], "enum": names},
            "s": {"type": ["string", "null"], "enum": ["left", "right", "center", None]},
            "p": {"type": ["string", "null"], "enum": ["nearest", None]},
            "k": {"type": "array", "items": {"type": "string"}},
            "r": {"type": ["string", "null"], "enum": ["room", "lab", "office", None]}},
            "required": ["i", "t", "x", "n", "s", "p", "k", "r"]}
        self.system = LLM_SYSTEM + "\nAllowed objects: " + ", ".join(self.classes)

    def _post(self, messages, num_predict=80, schema=None):
        body = json.dumps({"model": self.model, "stream": False, "think": False, "keep_alive": "24h",
                           "options": {"temperature": 0, "num_predict": num_predict},
                           "format": schema or self.schema, "messages": messages}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def warmup(self):
        t0 = time.perf_counter()
        try:
            old, self.timeout = self.timeout, 120.0   # first load can take ~30 s
            self.parse("find a chair")
            self.timeout = old
            print(f"LLM: {self.model} ready in {time.perf_counter() - t0:.1f} s (complex voice commands)")
            return True
        except Exception as e:  # noqa: BLE001
            self.timeout = old
            print(f"LLM: {self.model} unavailable ({e}); complex commands fall back to the rules")
            return False

    def parse(self, text):
        t0 = time.perf_counter()
        r = self._post([{"role": "system", "content": self.system}, {"role": "user", "content": text}])
        data = json.loads(r["message"]["content"])
        print(f"LLM: {text!r} -> {json.dumps(data, separators=(',', ':'))} ({time.perf_counter() - t0:.2f} s)")
        return data


PLACE_SYSTEM = """A blind user asked for an exit. From what the camera saw recently, decide where they are.
place: room (classroom, office, lab, meeting room, kitchen), restroom, hallway (corridor, lobby,
stairwell, entrance area) or unknown. reason: why, in under 12 words, spoken to the user
(e.g. "I see chairs, tables and a whiteboard"). Only use what is listed; if it is not enough, say unknown."""

PLACE_SCHEMA = {"type": "object", "properties": {
    "place": {"type": "string", "enum": ["room", "restroom", "hallway", "unknown"]},
    "reason": {"type": "string"}}, "required": ["place", "reason"]}


def _plan_place(self, question, scene_text):
    """One call per exit request: {"place": room|restroom|hallway|unknown, "reason": str} or None."""
    t0 = time.perf_counter()
    r = self._post([{"role": "system", "content": PLACE_SYSTEM},
                    {"role": "user", "content": f'Question: "{question}"\n{scene_text}'}],
                   num_predict=60, schema=PLACE_SCHEMA)
    d = json.loads(r["message"]["content"])
    print(f"LLM place: {json.dumps(d)} ({time.perf_counter() - t0:.2f} s)")
    if not isinstance(d, dict) or d.get("place") not in ("room", "restroom", "hallway", "unknown"):
        return None
    reason = re.sub(r"[^\w\s,.'-]", "", str(d.get("reason", "")))[:90].strip()
    return {"place": d["place"], "reason": reason}


LLMIntent.plan_place = _plan_place


def clean_llm(d, classes):
    """Validates the model's JSON; drops anything outside the allowed values."""
    classes = set(classes)
    if not isinstance(d, dict) or d.get("i") not in ("find", "read", "verify", "which", "cancel", "another", "other"):
        return None
    t = d.get("t") if d.get("t") in classes else None
    n = d.get("n") if d.get("n") in classes and d.get("n") != t else None
    x = [c for c in (d.get("x") or []) if c in classes and c not in (t, n)][:5]
    s = d.get("s") if d.get("s") in ("left", "right", "center") else None
    p = "nearest" if d.get("p") == "nearest" else None
    k = []
    for w in d.get("k") or []:
        w = " ".join(_words(str(w)))[:30]
        if w and w not in k:
            k.append(w)
    r = d.get("r") if d.get("r") in ("room", "lab", "office") else None
    return {"i": d["i"], "t": t, "x": x, "n": n, "s": s, "p": p, "k": k[:4], "r": r}
