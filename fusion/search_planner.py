"""Search planner for PLACES (restroom, room/lab numbers, named offices).

One planner per phone, started when a place is requested and stopped when it is found,
cancelled, or given up. It runs on the GB10 with data the server already has every frame
(detections + OCR through the target lock, LiDAR depth, ARKit pose) and sends spoken
"guide" messages to the phone. Obstacle alerts on the phone always take priority.

Stages
  scan         turn slowly all the way around here (12 slices of 30°, coverage.py)
  move         walk to the most open spot nearby (floor_map.best_spot on the shared 5 s
               floor map), room / unknown places only
  scan_center  turn all the way around again from there
  ask          "It isn't in this room. ... Should I take you to the door?"   (LLM once, search_llm.py)
  to_door      the target lock follows the room's door (exit_planner.exit_spec("room"))
  hall_scan    after the door: look around the hallway
  hall_walk    walk along the hallway reading signs; follow arrow signs; look around again now and then
  done         found / cancelled / gave up
Re-find mode (mode="refind", any target, started by place_search.py when the target has been out
of frame for 7 s):
  refind_scan  turn all the way around, starting toward the side the target was last seen (10 s)
  help         nothing found: "Help!" every 2 s until the target is found, cancelled, or 5 min pass
The moment the target itself is locked (sign or door text matches), the planner stops and the
normal find mode guides the user to it.
"""
import re

import numpy as np

from coverage import HeadingCoverage, heading_deg, signed_diff
from floor_map import best_spot, obstacle_ahead, open_distance
from space_shape import SpaceMap
from text_utils import text_matches

SAY_EVERY = 4.0        # s between spoken reminders in a stage
SCAN_TIMEOUT = 10.0    # s per look-around (full turn); no extension (was 30 s + 15 s)
HELP_EVERY = 2.0       # s between "Help!" after a re-find look-around found nothing
MIN_OPEN_M = 1.5       # fallback move: only toward a direction free for at least this far
MOVE_TIMEOUT = 35.0
ASK_REMIND = 12.0
ASK_GIVE_UP = 30.0
DOOR_TIMEOUT = 90.0
HALL_WALK_SCAN = 25.0  # s of walking before looking around again
TOTAL_TIMEOUT = 300.0
STEP_M = 0.7

YES = re.compile(r"^(?:yes|yeah|yep|sure|ok|okay|please|please do|go ahead|do it|lets go|alright|take me there)\b")
NO = re.compile(r"^(?:no|nope|not now|dont|do not|stay|keep looking|wait)\b")
ARROW_LEFT = re.compile(r"(?:←|<-|<|\bleft\b)")
ARROW_RIGHT = re.compile(r"(?:→|->|>|\bright\b)")


def _name(label):
    return label if label.startswith(("room ", "lab ", "office ")) else f"the {label}"


class SearchPlanner:
    def __init__(self, label, spec, place, lock, door_spec, llm=None, where_next=None, space=None,
                 mode="place", first_side="right"):
        self.label = label
        self.spec = spec              # the place target (sign/door + keywords)
        self.place = place            # room / restroom / hallway / unknown (at the request)
        self.lock = lock
        self.door_spec = door_spec    # target for "go to the door of this room"
        self.llm = llm
        self.where_next = where_next  # search_llm.where_next or None
        self.coverage = HeadingCoverage()
        # The shared 5 s LiDAR floor map (scene memory's). Without one (tests), keep our own.
        self.space = space if space is not None else SpaceMap()
        self.own_space = space is None
        self.mode = mode              # "place" (the stages above) or "refind" (look around, then help)
        self.stage = "refind_scan" if mode == "refind" else ("scan" if place != "hallway" else "hall_scan")
        self.started = None
        self.stage_started = None
        self.last_say = -1e9
        self.intro_done = False
        self.awaiting = False
        self.declines = 0
        self.goal = None
        self.turn_side = first_side if mode == "refind" else "right"   # keep turning one way
        self.open_by_bin = {}         # heading slice -> how far it's open (m), seen during look-arounds
        self.reminded = False
        self.move_mode = None         # "turn" / "walk" while moving to the open spot
        self.halfway_said = False
        self.frames = 0
        self.done = False
        self.searched = []            # for the where-next question

    # ------------------------------------------------------------------ helpers
    def _msg(self, say="", priority="guidance", **kw):
        return {"type": "guide", "say": say, "priority": priority, "stage": self.stage,
                "planner": "done" if self.done else "active", **kw}

    def _go(self, stage, now):
        self.stage = stage
        self.stage_started = now
        self.last_say = -1e9
        self.intro_done = False
        self.halfway_said = False
        if stage in ("scan", "scan_center", "hall_scan"):
            self.coverage.reset()
            self.turn_side = "right"

    def _due(self, now, every=SAY_EVERY):
        if now - self.last_say >= every:
            self.last_say = now
            return True
        return False

    def _finish(self, say, now, priority="reply"):
        self.done = True
        self.stage = "done"
        return [self._msg(say, priority)]

    def intro(self):
        """Added to the spoken reply when the search starts (so the first look-around step isn't
        said again right after it)."""
        self.intro_done = True
        return "Turn slowly to your right, all the way around."

    def stop(self):
        self.done = True
        self.stage = "done"

    # ------------------------------------------------------------------ per frame
    def update(self, frame, report, now, scene_summary):
        """Returns a list of guide messages for the phone."""
        if self.done:
            return []
        if self.started is None:
            self.started = self.stage_started = now
            if self.intro_done:
                self.last_say = now      # the reply just said "Turn slowly…": next reminder in 4 s
        self.frames += 1
        if self.own_space and self.mode != "refind":     # the re-find look-around needs no floor map
            self.space.add(frame, now)
        heading = heading_deg(frame.forward)
        pos = (float(frame.cam_pos[0]), float(frame.cam_pos[2]))

        # Found it: hand over to find mode (which announces "Found ...").
        if (self.stage != "to_door" and report and report.get("state") in ("acquired", "tracking")
                and report.get("label") == self.label):
            self.done = True
            self.stage = "done"
            return [self._msg("", "guidance", found=True)]
        if now - self.started > TOTAL_TIMEOUT:
            return self._finish(f"I couldn't find {_name(self.label)}. Say find {self.label} to try again.", now)

        handler = getattr(self, "_" + self.stage)
        if self.stage in ("scan", "scan_center"):
            d = open_distance(frame)
            if d is not None:
                b = int(heading // 30) % 12
                if d > self.open_by_bin.get(b, (0.0, 0.0))[0]:
                    self.open_by_bin[b] = (d, heading)   # farthest free view in this slice, and its exact heading
        return handler(frame, heading, pos, now, scene_summary) or []

    # ------------------------------------------------------------------ stages
    def _scan_stage(self, heading, now, next_stage, what="this area"):
        msgs = []
        new = self.coverage.update(heading, now)
        if new:
            msgs.append(self._msg(tap=True))
        n = self.coverage.count()
        waited = now - self.stage_started
        if n >= 11 or waited > SCAN_TIMEOUT:
            self.searched.append(f"looked all around {what} ({n} of 12 directions)")
            self._go(next_stage, now)
            return msgs
        if self._due(now):
            side = self.turn_side   # always the same way round, so the instructions never flip
            if n == 0 and not self.intro_done:
                if self.mode == "refind":
                    say = f"I lost {_name(self.label)}. Turn slowly to your {side}, all the way around."
                else:
                    say = f"Turn slowly to your {side}. I'm looking for {self.label} signs."
                self.intro_done = True
            elif n >= 6 and not self.halfway_said:
                say = "More than halfway around."
                self.halfway_said = True
            else:
                say = "Keep turning."                    # repeated: shorter
            msgs.append(self._msg(say))
        return msgs

    def _scan(self, frame, heading, pos, now, scene):
        nxt = "move" if self.place in ("room", "unknown", "restroom") else "ask"
        return self._scan_stage(heading, now, nxt)

    def _scan_center(self, frame, heading, pos, now, scene):
        return self._scan_stage(heading, now, "ask", "the middle of the room")

    def _move(self, frame, heading, pos, now, scene):
        if self.goal is None:
            self.goal = best_spot(self.space, pos) or self._open_direction_goal(pos)
            if self.goal is None:  # small room / nothing open: say so, then ask
                print("Planner: no open spot to move to")
                self.searched.append("there was no open space to move to")
                self._go("ask", now)
                return [self._msg("There isn't much open space to walk to here.", "reply")]
            print(f"Planner: moving to {self.goal}")
            self.searched.append("moved to the most open spot")
            self.move_mode = None
        dx, dz = self.goal["x"] - pos[0], self.goal["z"] - pos[1]
        dist = float(np.hypot(dx, dz))
        bearing = signed_diff(heading_deg((dx / max(dist, 1e-6), dz / max(dist, 1e-6))), heading)
        if dist < 0.7 or now - self.stage_started > MOVE_TIMEOUT:
            self._go("scan_center", now)
            self.last_say = now
            return [self._msg("Stop here. Now turn slowly all the way around again.", "reply")]
        # Turn until roughly facing the spot, then walk. Speak at once when that changes.
        side = "right" if bearing > 0 else "left"
        want = self.move_mode
        if want != "turn" and abs(bearing) > 35:
            want = "turn"
        elif want != "walk" and abs(bearing) < 20:
            want = "walk"
        changed = want != self.move_mode
        self.move_mode = want
        if not changed and not self._due(now):
            return []
        self.last_say = now
        if want == "turn":
            return [self._msg(f"Turn {side}." if self.intro_done else f"Turn slowly to your {side}.")]
        if obstacle_ahead(frame, 1.0):
            # Blocked on the way: this is as open as it gets from here. Look around from this spot.
            self._go("scan_center", now)
            self.last_say = now
            return [self._msg("Something is in front of you. Stop here and turn slowly all the way around.", "reply")]
        steps = max(1, int(round(dist / STEP_M)))
        where = "" if abs(bearing) < 12 else f", slightly {side}"
        if self.intro_done:                               # repeated: shorter
            return [self._msg(f"{steps} step{'s' if steps > 1 else ''} more{where}.")]
        self.intro_done = True
        return [self._msg(f"Let's move to a more open spot. Walk forward about {steps} step{'s' if steps > 1 else ''}{where}.")]

    def _open_direction_goal(self, pos):
        """Fallback when the floor map is too sparse: a point 1-3 m along the most open
        direction seen during the look-around."""
        if not self.open_by_bin:
            return None
        d, h = max(self.open_by_bin.values())
        if d < MIN_OPEN_M:
            return None
        h = np.radians(h)
        step = min(3.0, d - 1.0)
        return {"x": pos[0] + np.sin(h) * step, "z": pos[1] - np.cos(h) * step,
                "clearance_m": None, "travel_m": step, "from": "open direction"}

    def _ask(self, frame, heading, pos, now, scene):
        if not self.awaiting:
            self.awaiting = True
            self.reminded = False
            self.last_say = now
            reason = "It could be outside this room."
            if self.llm is not None and self.where_next is not None:
                try:
                    from scene_memory import SceneMemory
                    d = self.where_next(self.llm, self.label, SceneMemory.describe(scene), "; ".join(self.searched))
                    if d and d["next"] == "give_up":
                        return self._finish(f"{d['say']} I'll stop searching.", now)
                    if d and d["say"]:
                        reason = d["say"]
                except Exception as e:  # noqa: BLE001
                    print(f"LLM: where-next failed ({e})")
            return [self._msg(f"I don't see {_name(self.label)} here. {reason} "
                              "Should I take you to the door?", "reply", ask=True)]
        waited = now - self.stage_started
        if waited > ASK_GIVE_UP:
            self.awaiting = False
            return self._finish(f"I'll stop searching for now. Say find {self.label} when you want to try again.", now)
        if waited > ASK_REMIND and not self.reminded:
            self.reminded = True
            return [self._msg("Should I take you to the door? Say yes or no.", "reply", ask=True)]
        return []

    def _to_door(self, frame, heading, pos, now, scene):
        if now - self.stage_started > DOOR_TIMEOUT:
            return self._finish(f"I couldn't find the door. Say find {self.label} to try again.", now)
        return []  # the phone's find mode guides to the door and reports arrival

    def _refind_scan(self, frame, heading, pos, now, scene):
        return self._scan_stage(heading, now, "help", "here")

    def _help(self, frame, heading, pos, now, scene):
        """The look-around found nothing: call for help until the target is found or cancelled."""
        if self._due(now, HELP_EVERY):
            return [self._msg("Help!", "reply")]
        return []

    def _hall_scan(self, frame, heading, pos, now, scene):
        return self._scan_stage(heading, now, "hall_walk", "the hallway")

    def _hall_walk(self, frame, heading, pos, now, scene):
        if now - self.stage_started > HALL_WALK_SCAN:
            self._go("hall_scan", now)
            return [self._msg("Stop and turn slowly all the way around.", "reply")]
        if not self._due(now, 6.0):
            return []
        hint = self.arrow_hint(scene)
        if hint:
            return [self._msg(f"A sign points {hint}. Turn {hint} and walk slowly.")]
        if obstacle_ahead(frame, 1.2):
            return [self._msg("Turn slowly until the way ahead is clear.")]
        if self.intro_done:
            return [self._msg("Keep walking.")]          # repeated: shorter
        self.intro_done = True
        return [self._msg("Walk slowly forward along the hallway. I'm reading the signs.")]

    def arrow_hint(self, scene):
        """A sign that mentions the target and points left or right ("Restrooms ->")."""
        for text in (scene or {}).get("texts", []):
            if self.spec.get("keywords") and text_matches(text, self.spec["keywords"]):
                if ARROW_LEFT.search(text.lower()):
                    return "left"
                if ARROW_RIGHT.search(text.lower()):
                    return "right"
        return None

    # ------------------------------------------------------------------ events from the phone
    def on_answer(self, clean, now):
        """Yes/no after "Should I take you to the door?". Returns a spoken reply or None."""
        if not self.awaiting:
            return None
        if YES.match(clean):
            self.awaiting = False
            self._go("to_door", now)
            self.lock.find(self.door_spec)
            return "Okay. I'll take you to the door."
        if NO.match(clean):
            self.awaiting = False
            self.declines += 1
            if self.declines >= 2:
                self.done = True
                self.stage = "done"
                return f"Okay. Say find {self.label} when you want to try again."
            self._go("scan", now)
            return "Okay, I'll keep looking here. Turn slowly all the way around."
        return None

    def on_arrived(self, now):
        """The phone reached the door: back to the place target, then search the hallway."""
        self.lock.find(self.spec)
        self._go("hall_scan", now)
        self.last_say = now
        return [self._msg(f"Go through the door and stop. Then turn slowly; I'll look for {self.label} signs.",
                          "reply", find=self.label)]
