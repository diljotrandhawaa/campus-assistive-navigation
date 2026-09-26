"""Small text and speech helpers shared by the fusion server modules."""
import re

def text_words(text):
    t = (text or "").lower().replace("’", "").replace("'", "")
    return re.findall(r"[a-z0-9]+", t)


def clean_text(text):
    """Spoken command -> lowercase words without punctuation or apostrophes ("don't" -> "dont")."""
    t = (text or "").lower().replace("’", "").replace("'", "")
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def text_matches(text, keywords):
    """Whole-word match, so "women" does not match "men" and "204" does not match "2045"."""
    tokens = text_words(text)
    if not tokens:
        return False
    joined = f" {' '.join(tokens)} "
    for k in keywords:
        kt = " ".join(text_words(k))
        if kt and (f" {kt} " in joined):
            return True
    return False


def spoken_feet(meters):
    feet = meters * 3.28084
    if feet < 1:
        return "under 1 foot"
    n = int(round(feet))
    return "1 foot" if n == 1 else f"{n} feet"


def direction_words(bearing):
    if bearing is None:
        return "ahead"
    side = "left" if bearing < 0 else "right"
    a = abs(bearing)
    return ("straight ahead" if a < 8 else f"slightly {side}" if a < 25
            else f"to your {side}" if a < 60 else f"far {side}")


def with_article(label):
    return ("an " if label[:1] in "aeiou" else "a ") + label


def join_or(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]


def natural_join(items):
    items = [i for i in items if i]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def count_phrase(n, label):
    return f"a {label}" if n == 1 else f"{n} {label}s"


SIDE_THRESHOLD = 0.25    # |lateral| above this is "left"/"right" (m)


def side_text(lateral):
    if lateral is None:
        return None
    if lateral < -SIDE_THRESHOLD:
        return "left"
    if lateral > SIDE_THRESHOLD:
        return "right"
    return "ahead"


def target_phrase(label):
    return label if label.startswith("room ") else with_article(label)
