"""Glue between the server and the place search planner (one PlaceSearch per phone).

fusion_server.py only calls these hooks; all search logic lives in the separate modules:
    search_planner.py   stages: look around -> open spot -> look around -> ask -> door -> hallway
    coverage.py         which directions were already checked
    floor_map.py        LiDAR floor map, most open spot, obstacle ahead
    dynamic_prompts.py  extra YOLOE sign phrases for this search
    search_llm.py       the two one-time LLM questions
    scene_memory.py     where the user is (room / hallway / ...)

Hooks
    answer(clean)                     before the normal voice rules: yes/no to "take you to the door?"
    after_reply(text, reply)          after the voice rules: start / stop a search
    on_frame(report, now)             every frame, after the target lock -> guide messages
    on_target(action, reason, now)    phone's direct target control (cancel / arrived)
    close()                           phone disconnected
"""
import time

from dynamic_prompts import PromptManager, prompts_for
from exit_planner import exit_spec
from scene_memory import SceneMemory, classify_place
from search_planner import SearchPlanner

try:
    from search_llm import plan_target, where_next
except ImportError:  # pragma: no cover
    plan_target = where_next = None

NOT_PLACES = ("exit", "door")   # "find an exit" has its own planner (exit_planner.py)


def is_place_search(lock):
    spec = lock.spec
    return bool(spec and spec.get("keywords") and lock.label not in NOT_PLACES and not spec.get("goal"))


class PlaceSearch:
    def __init__(self, detector, lock, session, classes, llm=None):
        self.detector = detector
        self.lock = lock
        self.session = session
        self.classes = classes
        self.llm = llm
        self.prompts = PromptManager(detector)
        self.planner = None

    # ------------------------------------------------------------------ voice
    def answer(self, text, clean):
        """Yes/no while the planner is asking. Returns a voice_intent reply or None."""
        p = self.planner
        if p is None or p.done or not p.awaiting:
            return None
        say = p.on_answer(clean, time.monotonic())
        if say is None:
            return None
        base = {"type": "voice_intent", "transcript": text, "exclude": [], "side": None, "prefer": None,
                "candidates": [], "method": "planner", "say": say}
        if p.stage == "to_door":
            return {**base, "intent": "find", "target": self.lock.label}      # the phone guides to the door
        if p.done:  # declined twice
            self.stop()
            self.lock.cancel()
            return {**base, "intent": "cancel", "target": None}
        return {**base, "intent": "find", "target": p.label}

    def after_reply(self, text, reply):
        """Any new find / cancel ends the current search; a place find starts a new one."""
        intent = reply.get("intent")
        if intent not in ("find", "cancel") or reply.get("method") == "planner":
            return reply
        self.stop()
        if intent != "find" or not is_place_search(self.lock) or not self.session.get("ocr"):
            return reply
        self.start(text, reply)
        return reply

    # ------------------------------------------------------------------ search
    def start(self, text, reply):
        spec = dict(self.lock.spec)
        label = spec["label"]
        scene = self.session.get("scene")
        summary = scene.summary(time.monotonic()) if scene else {"frames": 0, "counts": {}, "texts": [],
                                                                 "hallway_views": 0, "views": 0}
        place, why = classify_place(summary)
        llm_prompts = []
        if self.llm is not None and plan_target is not None and summary["frames"] and place == "unknown":
            try:
                d = plan_target(self.llm, text, label, SceneMemory.describe(summary))
                if d:
                    place, llm_prompts = d["place"], d["prompts"]
            except Exception as e:  # noqa: BLE001
                print(f"LLM: search plan failed ({e}); using the rules")
        extra = self.prompts.apply(prompts_for(label, llm_prompts))
        if extra:
            # The new sign phrases become candidates of this target. For a restroom a restroom
            # pictogram counts on its own; for room numbers the text still has to match.
            spec["classes"] = set(spec["classes"]) | set(extra)
            if label == "restroom":
                spec["specific"] = set(spec["specific"]) | set(extra)
            self.lock.find(spec, side=self.lock.side, exclude=self.lock.exclude, prefer=self.lock.prefer)
        self.planner = SearchPlanner(label, spec, place, self.lock, exit_spec("room", self.classes),
                                     llm=self.llm, where_next=where_next if self.llm is not None else None)
        print(f"Place search: {label} | place={place} | extra prompts: {', '.join(extra) or 'none'}")
        reply["say"] = f"{reply.get('say', '')} {self.planner.intro()}".strip()
        reply["planner"] = self.planner.stage
        return reply

    def stop(self):
        if self.planner is not None:
            self.planner.stop()
            self.planner = None
        self.prompts.reset()

    # ------------------------------------------------------------------ per frame
    def on_frame(self, report, now):
        p = self.planner
        frame = self.session.get("last_frame")
        if p is None or p.done or frame is None:
            return []
        scene = self.session.get("scene")
        summary = scene.summary(now) if scene else {}
        msgs = p.update(frame, report, now, summary)
        if p.done:
            found = any(m.get("found") for m in msgs)
            self.planner = None
            if not found:           # gave up: end find mode too
                self.lock.cancel()
                self.prompts.reset()
                for m in msgs:
                    m["end_find"] = True
            # found: keep the lock and the extra prompts; normal find mode takes over
        return msgs

    def on_target(self, action, reason, now):
        """The phone's own target control. Returns guide messages."""
        p = self.planner
        if action == "cancel" and reason == "arrived" and p is not None and p.stage == "to_door":
            return p.on_arrived(now)
        if action in ("cancel", "find"):
            self.stop()
        return []

    def close(self):
        self.stop()
