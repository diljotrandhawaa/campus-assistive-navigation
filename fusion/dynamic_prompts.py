"""Extra YOLOE text prompts for one search, added and removed at runtime.

YOLOE is open-vocabulary: a place search can add phrases like "toilet sign" or "men's room sign"
so the detector looks for them too. They are removed when the search ends. Adding prompts only
re-encodes the text once (~tens of ms); per-frame cost is unchanged.

    pm = PromptManager(detector)          # detector.set_prompts(list) must exist
    added = pm.apply(prompts_for("restroom", llm_prompts))
    ...
    pm.reset()
"""
import re

BUILTIN = {
    "restroom": ["toilet sign", "men's room sign", "women's room sign", "wc sign",
                 "all gender restroom sign", "accessible restroom sign"],
    "room": ["room number sign", "door sign", "door plate"],
    "lab": ["room number sign", "door sign", "laboratory sign"],
    "office": ["room number sign", "door sign", "name plate"],
    "sign": ["door sign", "room number sign", "name plate"],
}
MAX_EXTRA = 8


def clean_prompt(p):
    p = re.sub(r"[^a-z' ]", " ", str(p).lower())
    p = " ".join(p.split())
    return p if 0 < len(p.split()) <= 4 else None


def prompts_for(label, llm_prompts=()):
    """Built-in synonyms for the kind of place, plus any the LLM suggested."""
    kind = label.split()[0] if label else ""
    base = BUILTIN.get(label) or BUILTIN.get(kind) or BUILTIN["sign"]
    out = []
    for p in list(base) + list(llm_prompts or ()):
        c = clean_prompt(p)
        if c and c not in out:
            out.append(c)
    return out[:MAX_EXTRA]


class PromptManager:
    def __init__(self, detector):
        self.detector = detector
        self.base = list(detector.classes)
        self.extra = []

    def apply(self, prompts):
        new = [p for p in prompts if p not in self.base][:MAX_EXTRA]
        if new != self.extra and hasattr(self.detector, "set_prompts"):
            self.detector.set_prompts(self.base + new)
            self.extra = new
            print(f"Prompts: added {', '.join(new) or 'nothing'}")
        return list(self.extra)

    def reset(self):
        if self.extra and hasattr(self.detector, "set_prompts"):
            self.detector.set_prompts(self.base)
            print(f"Prompts: removed {', '.join(self.extra)}")
        self.extra = []
