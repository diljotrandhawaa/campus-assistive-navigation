"""What the camera saw recently, and a rule-based guess of where the user is.

SceneMemory keeps the last 15 s: most objects of each class seen at once, sign text read by
OCR, and the LiDAR space hint (FrameData.space_hint). classify_place() turns that into
room / restroom / hallway / unknown plus spoken evidence ("I see a whiteboard and 6 chairs").
"""
from text_utils import count_phrase, natural_join, text_matches

class SceneMemory:
    """What the camera saw in the last WINDOW seconds: object counts (most seen at once),
    sign text, and the LiDAR space hint. Detection already sees every class each frame,
    so this costs nothing extra (no tracking)."""

    WINDOW = 15.0

    def __init__(self):
        self.frames = []   # (time, {label: count}, hint)
        self.texts = []    # (time, text)

    def add_frame(self, now, objects, hint):
        counts = {}
        for o in objects:
            counts[o["label"]] = counts.get(o["label"], 0) + 1
        self.frames.append((now, counts, hint))
        self.frames = [f for f in self.frames if now - f[0] <= self.WINDOW]

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
                "hallway_views": hallway, "views": len(hints)}

    @staticmethod
    def describe(summary):
        c = summary["counts"]
        seen = ", ".join(f"{n} {label}" for label, n in sorted(c.items(), key=lambda kv: -kv[1])) or "nothing"
        texts = "; ".join(summary["texts"][:6]) or "none"
        v = summary["views"]
        space = (f"walls on both sides with open space ahead in {summary['hallway_views']} of {v} views"
                 if v else "no depth information")
        return f"Seen in the last 15 s (most at once): {seen}. Signs read: {texts}. Space: {space}."


ROOM_THINGS = {"whiteboard": 2, "monitor": 1, "laptop": 1, "keyboard": 1, "printer": 1}


def classify_place(summary):
    """Rule-based guess: room / restroom / hallway / unknown, plus the evidence."""
    c = summary["counts"]
    if summary["frames"] < 5:
        return "unknown", "I've only just started looking"
    if c.get("toilet", 0) >= 1:
        return "restroom", "I see a toilet"
    room, why_room = 0, []
    for label, pts in ROOM_THINGS.items():
        if c.get(label, 0):
            room += pts
            why_room.append(count_phrase(c[label], label))
    if c.get("chair", 0) >= 3:
        room += 2
        why_room.append(f"{c['chair']} chairs")
    if c.get("table", 0) >= 2:
        room += 1
        why_room.append(f"{c['table']} tables")
    if c.get("sofa", 0) and c.get("table", 0):
        room += 1
    hall, why_hall = 0, []
    if summary["views"] >= 5 and summary["hallway_views"] / summary["views"] >= 0.4:
        hall += 2
        why_hall.append("a long open space with walls on both sides")
    if c.get("door", 0) >= 2:
        hall += 1
        why_hall.append(f"{c['door']} doors")
    if (c.get("exit sign", 0) or any(text_matches(t, ["exit"]) for t in summary["texts"])) and room == 0:
        hall += 1
        why_hall.append("an exit sign")
    if c.get("elevator", 0) or c.get("stairs", 0):
        hall += 1
        why_hall.append("an elevator or stairs")
    if room >= 2 and room > hall:
        return "room", "I see " + natural_join(why_room[:3])
    if hall >= 2 and hall > room:
        return "hallway", "I see " + natural_join(why_hall[:3])
    return "unknown", "I can't tell yet"
