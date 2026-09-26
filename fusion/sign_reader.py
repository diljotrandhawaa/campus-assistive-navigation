"""Reading signs aloud for a blind user ("read the sign", "which room is this?",
"is this room 204?").

The user can't aim the phone at a sign, so a request is not answered from one frame: the
server keeps reading (whole-frame OCR every READ_EVERY s) while the user turns, for up to
READ_WINDOW s, and answers with the first sign it finds, with where it is ("4 feet, to your
left"). If the first look finds nothing it says so once ("Turn slowly, I'll read the first
sign I find") and keeps looking. Nothing here ever asks the user to point at something.

    reading = SignReading(mode, now)             # mode: None | {"kind": "verify"|"which", ...}
    msg = reading.step(now, read_groups)         # read_groups() -> sign groups of this frame
    # msg: None (keep going) | {"say", "groups", "done"}
"""
import re

from text_utils import direction_words, spoken_feet, text_matches

READ_WINDOW = 8.0     # s of looking for a sign after a request
READ_EVERY = 0.6      # s between whole-frame OCR passes while looking
NO_SIGN = "I couldn't find a sign nearby."
LOOKING = "I don't see a sign yet. Turn slowly, and I'll read the first one I find."


def group_text_items(items):
    """Joins OCR lines that belong to the same sign (close together in the image)."""
    groups = []
    for it in sorted(items, key=lambda i: (i["box"][1], i["box"][0])):
        cx = (it["box"][0] + it["box"][2]) / 2
        for g in groups:
            gx = (g["box"][0] + g["box"][2]) / 2
            if abs(cx - gx) < 0.2 and it["box"][1] - g["box"][3] < 0.08:
                g["texts"].append(it["text"])
                g["box"] = [min(g["box"][0], it["box"][0]), min(g["box"][1], it["box"][1]),
                            max(g["box"][2], it["box"][2]), max(g["box"][3], it["box"][3])]
                g["near"] = g["near"] or (it.get("nearby_object") or {}).get("label")
                break
        else:
            groups.append({"texts": [it["text"]], "box": list(it["box"]),
                           "near": (it.get("nearby_object") or {}).get("label")})
    return groups


# "Lab 3B", "Room 204", "RM 12", or a bare 2-4 digit number like "204"
LABELED_NUMBER = re.compile(r"\b(room|rm|lab|laboratory|office|classroom|suite)\.?\s*#?\s*([a-z]?\d{1,4}[a-z]?)\b", re.I)
BARE_NUMBER = re.compile(r"\b([a-z]?\d{2,4}[a-z]?)\b", re.I)


def where_text(g):
    where = direction_words(g.get("bearing_deg"))
    return f"{spoken_feet(g['distance_m'])}, {where}" if g.get("distance_m") is not None else where


def answer_which(groups, room_kind=None):
    """"What room / lab is this?" -> the number on the nearest sign."""
    for g in groups:
        text = " ".join(g["texts"])
        m = LABELED_NUMBER.search(text)
        if m:
            kind = {"rm": "room", "laboratory": "lab"}.get(m.group(1).lower(), m.group(1).lower())
            number = m.group(2)
        else:
            m = BARE_NUMBER.search(text)
            if not m:
                continue
            kind, number = room_kind or "room", m.group(1)
        return f"This is {kind} {number.upper()}. The sign says: {text}. It's {where_text(g)}."
    if groups:
        return f"I can't find a number. The nearest sign says: {' '.join(groups[0]['texts'])}."
    return NO_SIGN


def answer_verify(groups, keywords, label=None):
    """"Is this the chemistry lab?" -> yes/no from the sign text."""
    if not groups:
        return NO_SIGN
    for g in groups:
        text = " ".join(g["texts"])
        if text_matches(text, keywords):
            return f"Yes. The sign says: {text}. It's {where_text(g)}."
    return f"I don't think so. The nearest sign says: {' '.join(groups[0]['texts'])}."


def describe_read(groups):
    if not groups:
        return NO_SIGN
    parts = []
    for g in groups[:3]:
        where = direction_words(g.get("bearing_deg"))
        if g.get("distance_m") is not None:
            where = f"{spoken_feet(g['distance_m'])}, {where}"
        thing = f"On the {g['near']}" if g.get("near") else "Sign"
        parts.append(f"{thing}, {where}: {'. '.join(g['texts'])}.")
    return " ".join(parts)


def answer(groups, mode):
    mode = mode or {}
    if mode.get("kind") == "verify":
        return answer_verify(groups, mode.get("keywords", []), mode.get("label"))
    if mode.get("kind") == "which":
        return answer_which(groups, mode.get("room_kind"))
    return describe_read(groups)


class SignReading:
    """One "read the sign" request, spread over several frames while the user turns."""

    def __init__(self, mode, now):
        self.mode = mode
        self.until = now + READ_WINDOW
        self.last = -1e9
        self.hinted = False

    def step(self, now, read_groups):
        if now - self.last < READ_EVERY:
            return None
        self.last = now
        groups = read_groups()
        if groups:
            return {"say": answer(groups, self.mode), "groups": groups, "done": True}
        if now >= self.until:
            return {"say": NO_SIGN, "groups": [], "done": True}
        if not self.hinted:
            self.hinted = True
            return {"say": LOOKING, "groups": [], "done": False}
        return None
