"""Find mode on the GB10: from a selected target to getting the user there.

Reused by every way of choosing a target (a spoken "find the chair", a place found through
its sign, "find an exit", the search planner's "take me to the door"): they all call
lock.find(spec), then lock.update(...) runs every frame and returns the report the phone
guides with.

1. Pick: among the detections of the spec's classes, the best candidate (sign text match >
   specific class > fallback), nearest first, optionally on one side / near something.
2. Lock: follow that BoT-SORT track id.
3. Anchor: once LiDAR measures it, the target also gets a fixed position in the room (ARKit
   world coordinates, + its width). From then on its distance and direction come from the
   phone's pose even when the detector loses it -- e.g. a door that no longer fits in the
   frame and is suddenly labelled "wall". Close up, the LiDAR depth in the anchor's direction
   gives the exact distance to its surface.
4. Rigid: while anchored, the lock is never dropped for time-outs; only a detection within
   about a metre of the anchor can take over (same object, new id). "Another one", a side
   change, cancel or a new find clears it. The phone ends it on arrival.
5. Objects detected at the target's position (the "wall" that is really the door) are marked
   "at_target" so the phone does not treat them as obstacles.

Report (sent as "target" with every result):
  label, state (acquired | tracking | lost_briefly | searching), track_id, matched, text, detail,
  distance_m, lateral_m, side, bearing_deg, anchored, anchor {x, y, z, width_m}
"""
import math

import numpy as np

from text_utils import side_text, target_phrase, text_matches

ANCHOR_BLEND = 0.4          # weight of a new measurement when refreshing the anchor
ANCHOR_SAME_M = 0.5         # a detection within half its width + this of the anchor is the same object
                            # (tight enough that the next door along the wall is NOT the same door)
ANCHOR_SAME_DEG = 12.0      # ... or, without depth, this close in direction
MIN_CONFIDENCE = 1
MAX_DEPTH = 5.5
FLOOR_MARGIN = 0.10


# ----------------------------------------------------------------------------- anchor geometry
def anchor_from(frame, o):
    """World position of a measured object (its near surface) and its width, or None."""
    d, l = o.get("distance_m"), o.get("lateral_m")
    if frame is None or d is None or l is None:
        return None
    x = frame.cam_pos[0] + frame.forward[0] * d + frame.right[0] * l
    z = frame.cam_pos[2] + frame.forward[1] * d + frame.right[1] * l
    u1, _, u2, _ = o["box"]
    width = (u2 - u1) * frame.h * d / frame.fy       # portrait u runs along the depth map's height
    return {"x": float(x), "y": float(frame.cam_pos[1]), "z": float(z),
            "width_m": float(min(3.0, max(0.3, width)))}


def _blend(old, new):
    if old is None or math.hypot(old["x"] - new["x"], old["z"] - new["z"]) > 1.0:
        return new
    w = ANCHOR_BLEND
    return {k: (1 - w) * old[k] + w * new[k] for k in ("x", "y", "z", "width_m")}


def _object_xz(frame, o):
    d, l = o.get("distance_m"), o.get("lateral_m")
    if frame is None or d is None or l is None:
        return None
    return (frame.cam_pos[0] + frame.forward[0] * d + frame.right[0] * l,
            frame.cam_pos[2] + frame.forward[1] * d + frame.right[1] * l)


def anchor_view(frame, anchor):
    """Distance ahead, sideways offset and direction of the anchor from the current pose.
    Close to and facing the anchor, the LiDAR depth in its direction refines the distance."""
    rel = np.array([anchor["x"] - frame.cam_pos[0], anchor["z"] - frame.cam_pos[2]])
    ahead, lateral = float(frame.forward @ rel), float(frame.right @ rel)
    bearing = math.degrees(math.atan2(lateral, ahead))
    source = "pose"
    if ahead > 0.15 and abs(bearing) < 25:
        near = _depth_toward(frame, anchor)
        if near is not None and abs(near - ahead) < 0.6:
            ahead, source = near, "lidar"
    return {"distance_m": round(ahead, 3), "lateral_m": round(lateral, 3),
            "bearing_deg": round(bearing, 1), "source": source}


def _depth_toward(frame, anchor):
    """Near-surface distance (along the walking direction) of the depth pixels around the
    anchor's projection in this frame, or None."""
    p = np.linalg.inv(frame.T) @ np.array([anchor["x"], anchor["y"], anchor["z"], 1.0])
    if p[2] > -0.1:                     # behind the camera
        return None
    px = frame.fx * p[0] / -p[2] + frame.cx
    py = frame.cy - frame.fy * p[1] / -p[2]
    half = max(4, int(anchor["width_m"] / 2 * frame.fx / -p[2] * 0.5))
    x0, x1 = int(max(0, px - half)), int(min(frame.w - 1, px + half))
    y0, y1 = int(max(0, py - 24)), int(min(frame.h - 1, py + 24))
    if x1 <= x0 or y1 <= y0:
        return None
    d = frame.depth[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
    c = frame.confidence[y0:y1 + 1, x0:x1 + 1]
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    ok = (c >= MIN_CONFIDENCE) & np.isfinite(d) & (d > 0.05) & (d < MAX_DEPTH)
    if ok.sum() < 8:
        return None
    d, xs, ys = d[ok], xs[ok], ys[ok]
    cam = np.stack([(xs + 0.5 - frame.cx) / frame.fx * d, -((ys + 0.5 - frame.cy) / frame.fy * d),
                    -d, np.ones_like(d)])
    w = frame.T @ cam
    keep = w[1] > frame.floor_y + FLOOR_MARGIN
    if keep.sum() < 8:
        return None
    ahead = frame.forward @ np.stack([w[0] - frame.cam_pos[0], w[2] - frame.cam_pos[2]])[:, keep]
    return float(np.percentile(ahead, 25))


# ----------------------------------------------------------------------------- the lock
class TargetLock:
    """Find mode: lock onto ONE tracked candidate of the target spec and report it every frame.

    Candidates: objects of the spec's classes. With keywords (places), a candidate whose
    OCR text matches is preferred; "specific" classes (e.g. "restroom sign") count on their
    own; generic ones (sign, door) only count when their text matches.
    """

    LOST_GRACE = 3.0  # seconds without an id update before searching again (built-in tracker)

    def __init__(self):
        self.cancel()

    def cancel(self):
        self.spec = None
        self.label = None
        self.side = None          # "left" / "center" / "right" / None
        self.exclude = set()
        self.prefer = "nearest"
        self.skip_ids = set()     # track ids rejected with "another one"
        self.track_id = None
        self.last_seen = None
        self.ever_seen = False
        self.generation = None
        self.smooth = None        # (distance, lateral) of the locked object, smoothed
        self.smooth_bearing = None
        self.anchor = None        # fixed room position of the locked object (see module doc)
        self.anchor_info = None   # what it was: matched label, text, detail, score

    def find(self, spec, side=None, exclude=(), prefer=None):
        self.cancel()
        self.spec = spec
        self.label = spec["label"]
        self.side = side
        self.exclude = set(exclude or ())
        self.prefer = prefer or "nearest"

    def another(self):
        if self.track_id is not None:
            self.skip_ids.add(self.track_id)
        self._unlock()

    def set_side(self, side):
        self.side = side
        self._unlock()

    def _unlock(self, keep_anchor=False):
        self.track_id = None
        self.last_seen = None
        self.smooth = None
        self.smooth_bearing = None
        if not keep_anchor:
            self.anchor = None
            self.anchor_info = None

    def _bearing(self, b):
        if b is None:
            return None
        if self.smooth_bearing is not None and abs(self.smooth_bearing - b) < 20:
            b = 0.5 * self.smooth_bearing + 0.5 * b
        self.smooth_bearing = b
        return round(b, 1)

    def _smoothed(self, d, l):
        if d is None or l is None:
            return d, l
        if self.smooth and abs(self.smooth[0] - d) < 0.8:
            d = 0.5 * self.smooth[0] + 0.5 * d
            l = 0.5 * self.smooth[1] + 0.5 * l
        self.smooth = (d, l)
        return round(d, 3), round(l, 3)

    @staticmethod
    def side_of(obj):
        s = obj.get("side")
        if s:
            return "center" if s == "ahead" else s
        u = (obj["box"][0] + obj["box"][2]) / 2
        return "left" if u < 1 / 3 else "right" if u > 2 / 3 else "center"

    # ------------------------------------------------------------------ candidates
    def _score(self, o):
        """2 = confirmed by sign text, 1 = counts by class alone, None = not a candidate."""
        if o["label"] not in self.spec["classes"] or o["label"] in self.exclude:
            return None
        if o.get("track_id") is None or o["track_id"] in self.skip_ids:
            return None
        if self.spec["keywords"] and text_matches(o.get("text"), self.spec["keywords"]):
            return 2
        if o["label"] in self.spec["specific"]:
            return 1
        return None

    @staticmethod
    def _by_clue(goal_obj, clue, within_m):
        """Is this door next to the clue (e.g. the exit sign)? Floor distance from LiDAR when
        both have depth, otherwise the sign sits above the door in the image."""
        if None not in (goal_obj.get("distance_m"), goal_obj.get("lateral_m"),
                        clue.get("distance_m"), clue.get("lateral_m")):
            return bool(np.hypot(goal_obj["distance_m"] - clue["distance_m"],
                                 goal_obj["lateral_m"] - clue["lateral_m"]) <= within_m)
        gx1, gy1, gx2, gy2 = goal_obj["box"]
        cx = (clue["box"][0] + clue["box"][2]) / 2
        pad = (gx2 - gx1) * 0.3
        return gx1 - pad <= cx <= gx2 + pad and clue["box"][3] <= gy1 + (gy2 - gy1) * 0.3 \
            and clue["box"][3] >= gy1 - 0.25

    def _scores(self, objects):
        """[(score, obj)]: 3 = goal (door by the clue / door with the text), 2 = text match,
        1 = specific class, 0.5 = fallback class (e.g. any door when the place is unknown)."""
        spec = self.spec
        goal = spec.get("goal")
        fallback = spec.get("fallback") or set()
        base = [(self._score(o), o) for o in objects]
        clues = [o for sc, o in base if sc is not None and not (goal and o["label"] in goal["classes"])]
        out = []
        for sc, o in base:
            usable = o.get("track_id") is not None and o["track_id"] not in self.skip_ids
            o.pop("via", None)
            if goal and usable and o["label"] in goal["classes"]:
                if sc == 2:
                    sc = 3
                else:
                    clue = next((c for c in clues if self._by_clue(o, c, goal["near_clue_m"])), None)
                    if clue is not None:
                        sc = 3
                        o["via"] = clue["label"]
            if sc is None and usable and o["label"] in fallback:
                sc = 0.5
            out.append((sc, o))
        return out

    def _pick(self, pool):
        if self.prefer == "nearest":
            return min(pool, key=lambda o: (o["distance_m"] is None, o["distance_m"] or 0, -o["confidence"]))
        return max(pool, key=lambda o: o["confidence"])

    def _near_ok(self, o, objects):
        """"the sofa near the window": another object of a `near` class within `within_m`."""
        near = self.spec.get("near")
        if not near:
            return True
        if o.get("distance_m") is None or o.get("lateral_m") is None:
            return False
        for other in objects:
            if (other is not o and other["label"] in near["classes"]
                    and other.get("distance_m") is not None and other.get("lateral_m") is not None
                    and np.hypot(other["distance_m"] - o["distance_m"],
                                 other["lateral_m"] - o["lateral_m"]) <= near["within_m"]):
                return True
        return False

    def _at_anchor(self, o, frame):
        """Is this detection the anchored object (same place in the room)?"""
        if self.anchor is None:
            return False
        xz = _object_xz(frame, o)
        if xz is not None:
            reach = max(ANCHOR_SAME_M, self.anchor["width_m"] / 2 + 0.15)
            return math.hypot(xz[0] - self.anchor["x"], xz[1] - self.anchor["z"]) <= reach
        b = o.get("bearing_deg")
        if frame is None or b is None:
            return False
        return abs(b - anchor_view(frame, self.anchor)["bearing_deg"]) <= ANCHOR_SAME_DEG

    def _mark_at_target(self, objects, frame, target):
        """Detections sitting on the target (the door seen up close as a "wall") are part of it."""
        if self.anchor is None or frame is None:
            return
        reach = max(ANCHOR_SAME_M, self.anchor["width_m"] / 2 + 0.15)
        for o in objects:
            if o is target or o["label"] in self.spec["classes"]:
                continue   # another candidate (e.g. the next door) is never "part of" the target
            xz = _object_xz(frame, o)
            if xz is not None and math.hypot(xz[0] - self.anchor["x"], xz[1] - self.anchor["z"]) <= reach:
                o["at_target"] = True

    # ------------------------------------------------------------------ reports
    def _detail(self, o):
        """What was actually found, when it isn't simply the target class."""
        if o["label"] == self.label and not o.get("text"):
            return None
        if o.get("via"):
            return f'{o["label"]} by the {o["via"]}'
        if o.get("text"):
            return f'{o["label"]} that says "{o["text"][:40]}"'
        return o["label"]

    def _anchor_fields(self, from_anchor=False):
        """anchored = this report comes from the room position (the detector can't see it now)."""
        if self.anchor is None:
            return {"anchored": False}
        a = self.anchor
        return {"anchored": from_anchor, "anchor": {"x": round(a["x"], 3), "y": round(a["y"], 3),
                                             "z": round(a["z"], 3), "width_m": round(a["width_m"], 2)}}

    def _report(self, state, o, frame=None):
        # Refresh the anchor from this detection (it is the locked object).
        new = anchor_from(frame, o)
        if new is not None:
            self.anchor = _blend(self.anchor, new)
            self.anchor_info = {"matched": o["label"], "text": o.get("text"), "detail": self._detail(o)}
        d, l = self._smoothed(o["distance_m"], o["lateral_m"])
        report = {"label": self.label, "side_filter": self.side, "state": state, "track_id": o["track_id"],
                  "matched": o["label"], "text": o.get("text"), "detail": self._detail(o),
                  "distance_m": d, "lateral_m": l, "side": side_text(l),
                  "bearing_deg": self._bearing(o.get("bearing_deg")), **self._anchor_fields()}
        return report

    def _anchored_report(self, frame):
        """The detector can't see the target right now: report it from its room position."""
        v = anchor_view(frame, self.anchor)
        d, l = self._smoothed(v["distance_m"], v["lateral_m"])
        info = self.anchor_info or {}
        return {"label": self.label, "side_filter": self.side, "state": "tracking", "track_id": self.track_id,
                "matched": info.get("matched"), "text": info.get("text"), "detail": info.get("detail"),
                "distance_m": d, "lateral_m": l, "side": side_text(l),
                "bearing_deg": self._bearing(v["bearing_deg"]), "source": v["source"], **self._anchor_fields(True)}

    def _lock_on(self, pick, now, frame, state="acquired"):
        if state == "acquired":
            self.anchor = None           # a different object: start a new anchor
            self.anchor_info = None
        self.track_id, self.last_seen, self.ever_seen = pick["track_id"], now, True
        self.smooth = None
        self.smooth_bearing = None
        pick["is_target"] = True
        return self._report(state, pick, frame)

    # ------------------------------------------------------------------ per frame
    def update(self, objects, now, tracking=None, frame=None):
        """tracking: CameraTracker info {"generation","alive_ids","retention_seconds"} or None.
        frame: this frame's FrameData (pose + depth) for the anchor; None = no anchor."""
        if not self.spec:
            return None
        report = self._update(objects, now, tracking, frame)
        target = next((o for o in objects if o.get("is_target")), None)
        self._mark_at_target(objects, frame, target)
        return report

    def _update(self, objects, now, tracking, frame):
        base = {"label": self.label, "side_filter": self.side}
        if tracking is not None and tracking.get("generation") != self.generation:
            # Tracker rebuilt (new target set, image size, long gap): ids are meaningless, the anchor isn't.
            self.generation = tracking.get("generation")
            self.skip_ids.clear()
            self._unlock(keep_anchor=True)

        if self.track_id is not None:
            for o in objects:
                if o.get("track_id") == self.track_id:
                    if self.spec.get("goal"):
                        # Locked on the clue (exit sign)? Switch to the goal (the door by it) once seen.
                        scored = self._scores(objects)
                        current = next((sc for sc, x in scored if x is o), None) or 0
                        better = [x for sc, x in scored if sc == 3 and x is not o
                                  and (self.side is None or self.side_of(x) == self.side)]
                        if current < 3 and better:
                            return self._lock_on(self._pick(better), now, frame)
                    self.last_seen = now
                    o["is_target"] = True
                    return self._report("tracking", o, frame)
            if tracking is not None:
                still_alive = (self.track_id in tracking.get("alive_ids", [])
                               and now - self.last_seen <= tracking.get("retention_seconds", 10.0))
            else:
                still_alive = now - self.last_seen <= self.LOST_GRACE
            if not still_alive:
                self._unlock(keep_anchor=True)  # the id is gone; the anchor (if any) stays
            elif self.anchor is None or frame is None:
                return {**base, "state": "lost_briefly", "track_id": self.track_id}

        scored = self._scores(objects)
        eligible = [(sc, o) for sc, o in scored
                    if sc is not None and (self.side is None or self.side_of(o) == self.side)
                    and self._near_ok(o, objects)]

        if self.anchor is not None and frame is not None:
            # Rigid lock: only the same object (at the anchor) may take over, or a better goal.
            same = [o for sc, o in eligible if self._at_anchor(o, frame)]
            if self.spec.get("goal"):
                goals = [o for sc, o in eligible if sc == 3 and not self._at_anchor(o, frame)]
                if goals and (self.anchor_info or {}).get("matched") not in self.spec["goal"]["classes"]:
                    return self._lock_on(self._pick(goals), now, frame)
            if same:
                return self._lock_on(self._pick(same), now, frame, state="tracking")
            return self._anchored_report(frame)

        if not eligible:
            return {**base, "state": "searching", "seen_before": self.ever_seen}
        best = max(sc for sc, _ in eligible)  # text-confirmed candidates win
        pool = [o for sc, o in eligible if sc == best]
        return self._lock_on(self._pick(pool), now, frame)

    def status_text(self):
        if not self.label:
            return "Not searching for anything."
        where = f" on your {self.side}" if self.side in ("left", "right") else ""
        return f"Looking for {target_phrase(self.label)}{where}."
