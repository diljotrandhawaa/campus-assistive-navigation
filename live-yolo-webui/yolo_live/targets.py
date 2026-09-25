"""Shared detector vocabulary and conservative English command matching."""

import re
import unicodedata


CLASSES = (
    "table",
    "door",
    "glass door",
    "wooden door",
    "sliding door",
    "revolving door",
    "trapdoor",
    "door handle",
    "garage door",
    "brown door",
    "green door",
    "white door",
    "office door",
    "classroom door",
    "doorknob",
    "door lever",
    "door latch",
    "pull handle",
    "push handle",
    "push bar",
    "panic bar",
    "door pull",
    "door hardware",
    "door lock cylinder",
    "handicap push button",
    "elevator",
    "elevator door",
    "chair",
    "whiteboard",
    "sign",
    "stairs",
    "person",
    "backpack",
    "cardboard box",
    "trash can",
    "sofa",
    "bench",
    "window",
    "toilet",
    "monitor",
    "laptop",
    "keyboard",
    "printer",
)

ALIASES = {label: label for label in CLASSES}
ALIASES.update({
    "seat": "chair",
    "chairs": "chair",
    "tables": "table",
    "white board": "whiteboard",
    "whiteboards": "whiteboard",
    "doors": "door",
    "door knob": "doorknob",
    "door handles": "door handle",
    "handle door": "door handle",
    "revolved door": "revolving door",
    "lift": "elevator",
    "lift door": "elevator door",
    "bin": "trash can",
    "trash bin": "trash can",
    "rubbish bin": "trash can",
    "garbage can": "trash can",
    "wastebasket": "trash can",
    "couch": "sofa",
    "staircase": "stairs",
    "stairway": "stairs",
    "computer monitor": "monitor",
    "laptop computer": "laptop",
    "cardboard carton": "cardboard box",
})

# These words do not change which physical object is requested.
FILLER = set("""
    please find locate show me take guide to toward towards
    i want would like need go get see a an the my on at
    object target can could you help looking for
""".split())

# These require interpretation beyond this deliberately limited grammar.
COMPLEX = {
    "not", "no", "dont", "don't", "instead", "actually", "except",
    "avoid", "without", "rather", "neither", "nor", "but",
}

SIDES = {
    "left": "left",
    "right": "right",
    "center": "center",
    "centre": "center",
    "middle": "center",
}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'").replace("'", "")
    return re.sub(r"[^\w\s]", " ", text)


PHRASES = sorted(
    ((tuple(normalize(alias).split()), target)
     for alias, target in ALIASES.items()),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def interpret_text(transcript):
    """Return a target request, never a visual acquisition claim."""
    result = {
        "transcript": transcript,
        "status": "needs_clarification",
        "intent": None,
        "target": None,
        "qualifiers": {"horizontal": None},
        "message": "",
    }

    def reject(status, message):
        return {**result, "status": status, "message": message}

    words = normalize(transcript).split()
    if not words:
        return reject("no_speech", "No speech was recognized.")

    if len(words) > 40 or len(transcript) > 400:
        return reject(
            "needs_clarification",
            "Please use a short command naming one object.",
        )

    if set(words) & COMPLEX:
        return reject(
            "needs_clarification",
            "Please repeat only the object you want, for example: "
            "'Find the table on the left'.",
        )

    # Longest match prevents "door" from swallowing "door handle".
    targets = []
    remaining = []
    index = 0
    while index < len(words):
        for phrase, target in PHRASES:
            if tuple(words[index:index + len(phrase)]) == phrase:
                targets.append(target)
                index += len(phrase)
                break
        else:
            remaining.append(words[index])
            index += 1

    if not targets:
        return reject(
            "unsupported_target",
            "No supported object name was recognized. "
            "Use an object from the target list.",
        )

    if len(targets) != 1:
        return reject(
            "needs_clarification",
            "Please name one target object per command.",
        )

    sides = {SIDES[word] for word in remaining if word in SIDES}
    if len(sides) > 1:
        return reject(
            "needs_clarification",
            "Please choose one side: left, center, or right.",
        )

    unsupported = [
        word for word in remaining
        if word not in FILLER and word not in SIDES
    ]
    if unsupported:
        return reject(
            "needs_clarification",
            "I understood the object but not the full command. "
            "Try 'Find the chair' or 'Find the chair on the left'.",
        )

    return {
        **result,
        "status": "resolved",
        "intent": "select_target",
        "target": targets[0],
        "qualifiers": {
            "horizontal": next(iter(sides), None),
        },
        "message": "Target request understood.",
    }