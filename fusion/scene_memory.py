"""What the camera saw recently (data only; where the user is comes from place_vlm.py).

SceneMemory keeps the last 5 s: most objects of each class seen at once, sign text read by
OCR, the LiDAR space hint (FrameData.space_hint), where doors were seen, and the shape of the
free space around the user (space_shape.SpaceMap). describe() turns it into a short text that
the VLM gets as hints, and the search planner uses the floor map.
"""
import numpy as np

from space_shape import SpaceMap

# Door-like classes (the detector has many door words); door parts don't count as doors.
DOOR_PARTS = ("handle", "knob", "lever", "latch", "lock", "pull", "hardware", "trapdoor", "push bar",
              "panic bar", "button")


def is_door(label):
    return "door" in label and not any(p in label for p in DOOR_PARTS)

class SceneMemory:
    """What the camera saw in the last WINDOW seconds: object counts (most seen at once),
    sign text, and the LiDAR space hint. Detection already sees every class each frame,
    so this costs nothing extra (no tracking)."""

    WINDOW = 5.0

    def __init__(self):
        self.frames = []   # (time, {label: count}, hint)
        self.texts = []    # (time, text)
        self.doors = []    # (time, x, z) room positions of doors seen with depth
        self.space = SpaceMap()

    def add_frame(self, now, objects, hint, frame=None):
        counts = {}
        for o in objects:
            counts[o["label"]] = counts.get(o["label"], 0) + 1
        self.frames.append((now, counts, hint))
        self.frames = [f for f in self.frames if now - f[0] <= self.WINDOW]
        if frame is not None:
            self.space.add(frame, now)
            for o in objects:
                if is_door(o["label"]) and o.get("distance_m") is not None and o.get("lateral_m") is not None:
                    self.doors.append((now, *frame.to_world(o["distance_m"], o["lateral_m"])))
            self.doors = [d for d in self.doors if now - d[0] <= self.WINDOW][-200:]

    def add_text(self, now, text):
        if text:
            self.texts.append((now, text))
            self.texts = [t for t in self.texts if now - t[0] <= self.WINDOW][-20:]

    def summary(self, now):
        frames = [f for f in self.frames if now - f[0] <= self.WINDOW]
        counts = {}
        for _, c, _ in frames:
            for label, n in c.items():
                counts[label] = max(counts.get(label, 0), n)
        hints = [h for _, _, h in frames if h]
        hallway = sum(1 for h in hints if h["open_ahead"] and h["wall_left"] and h["wall_right"])
        texts = list(dict.fromkeys(t for tt, t in self.texts if now - tt <= self.WINDOW))
        return {"frames": len(frames), "counts": counts, "texts": texts,
                "hallway_views": hallway, "views": len(hints),
                "shape": self.space.shape(now), "door_spots": self._door_spots(now)}

    def _door_spots(self, now):
        """Distinct doors seen (room positions merged within 0.8 m)."""
        spots = []
        for t, x, z in self.doors:
            if now - t > self.WINDOW:
                continue
            if all(np.hypot(x - sx, z - sz) > 0.8 for sx, sz in spots):
                spots.append((x, z))
        return spots

    @staticmethod
    def describe(summary):
        c = summary["counts"]
        seen = ", ".join(f"{n} {label}" for label, n in sorted(c.items(), key=lambda kv: -kv[1])) or "nothing"
        texts = "; ".join(summary["texts"][:6]) or "none"
        v = summary["views"]
        space = (f"walls on both sides with open space ahead in {summary['hallway_views']} of {v} views"
                 if v else "no depth information")
        shape = summary.get("shape")
        if shape:
            space += (f"; free area about {shape['length_m']} m long and {shape['width_m']} m wide"
                      f"{', walls along both long sides' if shape['walls_both_sides'] else ''}")
        return f"Seen in the last {SceneMemory.WINDOW:.0f} s (most at once): {seen}. Signs read: {texts}. Space: {space}."
