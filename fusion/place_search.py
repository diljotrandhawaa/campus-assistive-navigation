"""Glue between the server and the place search planner (one PlaceSearch per phone).

fusion_server.py only calls these hooks; all search logic lives in the separate modules:
    search_planner.py   stages: look around -> open spot -> look around -> ask -> door -> hallway
    coverage.py         which directions were already checked
    floor_map.py        LiDAR floor map, most open spot, obstacle ahead
    dynamic_prompts.py  extra YOLOE sign phrases for this search
    search_llm.py       the two one-time LLM questions
    place_vlm.py        where the user is (room / hallway / ...), from the camera model

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
from place_vlm import scene_hints
from search_planner import SearchPlanner

try:
    from search_llm import where_next
except ImportError:  # pragma: no cover
    where_next = None

PLACE_WAIT_S = 4.0   # wait this long for a fresh VLM answer when the last one is old

NOT_PLACES = ("exit", "door")   # "find an exit" has its own planner (exit_planner.py)
REFIND_AFTER_S = 7.0            # target out of frame this long (and no fixed room spot): look around


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
        self.out_since = None         # when the target was last in view (re-find timer)
        self.last_side = "right"      # side it was last seen on
        self.watch_label = None

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
        now = time.monotonic()
        ctx = self.session.get("place")
        p = ctx.get(now, lambda: scene_hints(scene, now), wait=PLACE_WAIT_S) if ctx is not None else None
        place = p["planner_place"] if p else "unknown"
        extra = self.prompts.apply(prompts_for(label, []))
        if extra:
            # The new sign phrases become candidates of this target. For a restroom a restroom
            # pictogram counts on its own; for room numbers the text still has to match.
            spec["classes"] = set(spec["classes"]) | set(extra)
            if label == "restroom":
                spec["specific"] = set(spec["specific"]) | set(extra)
            self.lock.find(spec, side=self.lock.side, exclude=self.lock.exclude, prefer=self.lock.prefer)
        self.planner = SearchPlanner(label, spec, place, self.lock, exit_spec("room", self.classes),
                                     llm=self.llm, where_next=where_next if self.llm is not None else None,
                                     space=scene.space if scene is not None else None)
        print(f"Place search: {label} | place={place} ({p['place'] if p else 'no VLM'}) | extra prompts: {', '.join(extra) or 'none'}")
        first = (reply.get("say") or "").split(". ")[0].rstrip(".")      # "Looking for a restroom"
        reply["say"] = f"{first}. {self.planner.intro()}" if first else self.planner.intro()
        reply["planner"] = self.planner.stage
        return reply

    def stop(self):
        if self.planner is not None:
            self.planner.stop()
            self.planner = None
        self.out_since = time.monotonic()
        self.prompts.reset()

    # ------------------------------------------------------------------ per frame
    def _watch(self, report, now):
        """Re-find: the target has been out of frame for REFIND_AFTER_S (and has no fixed room spot,
        so the phone can't point to it) -> start a look-around (search_planner mode="refind").
        Targets LiDAR has measured stay "tracking" from their anchor and never trigger this."""
        label = self.lock.label if self.lock.spec else None
        if label != self.watch_label:            # new target / cancelled: restart the timer
            self.watch_label, self.out_since = label, now
        if label is None or self.planner is not None or report is None:
            return
        if report.get("state") in ("acquired", "tracking"):
            self.out_since = now
            b = report.get("bearing_deg")
            if b is not None and abs(b) > 5:
                self.last_side = "left" if b < 0 else "right"
            return
        if now - self.out_since < REFIND_AFTER_S:
            return
        scene = self.session.get("scene")
        self.planner = SearchPlanner(label, dict(self.lock.spec), "refind", self.lock, None,
                                     space=scene.space if scene is not None else None,
                                     mode="refind", first_side=self.last_side)
        print(f"Re-find: {label} out of frame {now - self.out_since:.0f} s -> look around, "
              f"starting {self.last_side}")

    def on_frame(self, report, now):
        self._watch(report, now)
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
            self.out_since = now
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
