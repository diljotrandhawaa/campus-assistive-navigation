"""Hybrid rule-based and semantic voice target selection.

Public interface:
    warmup_command_matcher()
    interpret_command(transcript)

Sentence embeddings rank requests against configured detector classes.
They do not verify visibility, distance, or physical object identity.

The language guards in this version are English.
"""

import logging
import math
import os
import re
import threading
import unicodedata

from .direction import CLASSES


LOG = logging.getLogger("yolo_live.voice.commands")

MODEL_ID = os.environ.get(
    "VOICE_MATCH_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
MODEL_DEVICE = os.environ.get("VOICE_MATCH_DEVICE", "cpu")

# Starting values, not calibrated probabilities.
MIN_SIMILARITY = float(os.environ.get("VOICE_MATCH_MIN_SCORE", "0.65"))
MIN_MARGIN = float(os.environ.get("VOICE_MATCH_MIN_MARGIN", "0.08"))

SUPPORTED = tuple(dict.fromkeys(CLASSES))
REJECT = "__reject__"

_MODEL_LOCK = threading.Lock()
_MATCHER = None


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'").replace("'", "")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

def strip_politeness(text):
    # Remove greetings and leading politeness, preserving the actual request.
    text = re.sub(
        r"^(?:(?:hi|hello|hey)(?: there)?\s+|please\s+)+",
        "",
        text,
    )

    # Remove trailing conversational phrases.
    text = re.sub(
        r"(?:\s+(?:for me|please|thanks|thank you))+$",
        "",
        text,
    )

    return text.strip()

ALIASES = {normalize(label): label for label in SUPPORTED}

for alias, target in {
    "chairs": "chair",
    "tables": "table",
    "white board": "whiteboard",
    "doors": "door",
    "door knob": "doorknob",
    "door handles": "door handle",
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
    "restroom": "restroom sign",
    "bathroom": "restroom sign",
    "washroom": "restroom sign",
}.items():
    if target in SUPPORTED:
        ALIASES.setdefault(normalize(alias), target)

_NAMES = "|".join(
    re.escape(name)
    for name in sorted(ALIASES, key=len, reverse=True)
)

# Longest phrases come first: "door handle" before "door".
MENTION = re.compile(rf"(?<!\w)(?:{_NAMES})(?!\w)")

# These simple, explicit commands do not need model inference.
DIRECT_COMMAND = re.compile(
    r"(?:please )?"
    r"(?:(?:find|locate|show me|take me to|guide me to|"
    r"can you find|could you find|"
    r"i want to go to|i want to find|i need|look for|"
    r"sit in|sit on|i want to sit in|i want to sit on) )?"
    r"(?:a |an |the )?"
    rf"(?P<object>{_NAMES})"
    r"(?: please)?"
)

# Do not let semantic similarity override negation or corrections.
COMPLEX_COMMAND = re.compile(
    r"\b(?:no|not|never|dont|cannot|cant|wont|shouldnt|"
    r"avoid|without|except|unless|instead|actually|rather|"
    r"stop|cancel|ignore|forget|neither|nor|and|or|then)\b"
)

SIDE_PHRASE = re.compile(
    r"\b(?:on|to) (?:(?:my|the) )?"
    r"(?P<side>left|right|center|centre|middle)"
    r"(?: (?:hand )?side)?\b"
)

# These require vision capabilities beyond class and horizontal position.
# Apply this after masking recognized class names, so "white door"
# remains valid when it is an actual configured class.
UNSUPPORTED_QUALIFIER = re.compile(
    r"\b(?:nearest|closest|farthest|furthest|near|beside|behind|between|"
    r"next to|in front of|first|second|third|last|leftmost|rightmost|"
    r"red|blue|yellow|black|white|green|brown|purple|orange|"
    r"big|small|large|tall|short|wooden|glass|"
    r"left|right|center|centre|middle)\b"
)

GENERIC_SEATING = re.compile(r"\b(?:sit|sitting|seat|seating|rest)\b")


# Add representative requests here to improve coverage.
# Keep them tied to the meaning of the class, not arbitrary associations.
EXAMPLES = {
    "chair": [
        "I would like to sit in a chair.",
        "Help me get to that chair.",
        "Could you point me toward the chair?",
    ],
    "trash can": [
        "Where can I throw this away?",
        "I need somewhere to dispose of this rubbish.",
        "Help me find a waste receptacle.",
        "Where should I put this garbage?",
    ],
    "elevator": [
        "Show me where the lift is.",
        "I want to take the elevator.",
        "Help me get to the lift.",
    ],
    "door": [
        "Help me find the doorway.",
        "Show me the door.",
        "I want to approach that door.",
    ],
    "whiteboard": [
        "Show me the board used for writing with markers.",
        "Help me find the whiteboard.",
    ],
    "sofa": [
        "Help me get to the couch.",
        "I want to sit on the sofa.",
    ],
    "bench": [
        "I would like to sit on the bench.",
        "Help me find a bench.",
    ],
    "toilet": [
        "Help me locate the toilet.",
        "Show me the toilet.",
    ],
    "stairs": [
        "Help me find the staircase.",
        "I want to take the stairs.",
    ],
    "printer": [
        "Show me the machine used to print documents.",
        "Where can I collect my printed pages?",
    ],
}

# Competing examples help reject unrelated statements and unsupported
# destinations. They cannot cover every possible out-of-scope request.
REJECT_EXAMPLES = [
    "Hello, how are you?",
    "Thank you for your help.",
    "What time is it?",
    "Tell me a joke.",
    "What is a chair?",
    "The chair is broken.",
    "I fell off a chair.",
    "The door is closed.",
    "I already found what I needed.",
    "I do not want to select an object.",
]

for destination in ("kitchen", "bedroom", "cafeteria", "room 204"):
    if destination not in SUPPORTED:
        REJECT_EXAMPLES.append(f"Take me to the {destination}.")


def response(
    transcript,
    status,
    message,
    target=None,
    horizontal=None,
    method=None,
    **details,
):
    return {
        "transcript": transcript,
        "status": status,
        "target": target,
        "horizontal": horizontal,
        "message": message,
        "method": method,
        **details,
    }


def clarify(transcript, message, **details):
    return response(
        transcript, "needs_clarification", message, **details,
    )


class SemanticMatcher:
    def __init__(self):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if not SUPPORTED:
            raise ValueError("The detector class list is empty.")

        if (
            not math.isfinite(MIN_SIMILARITY)
            or not 0 <= MIN_SIMILARITY <= 1
            or not math.isfinite(MIN_MARGIN)
            or not 0 <= MIN_MARGIN <= 2
        ):
            raise ValueError("Invalid voice matching thresholds.")

        self.np = np
        LOG.info(
            "Loading voice command matcher: %s on %s",
            MODEL_ID, MODEL_DEVICE,
        )
        self.model = SentenceTransformer(
            MODEL_ID,
            device=MODEL_DEVICE,
            trust_remote_code=False,
        )

        texts = []
        self.labels = []

        for label in SUPPORTED:
            examples = [
                f"Find the {label}.",
                f"Take me to the {label}.",
                f"I want to go to the {label}.",
                f"Help me locate the {label}.",
                *EXAMPLES.get(label, []),
            ]
            for text in dict.fromkeys(examples):
                texts.append(text)
                self.labels.append(label)

        for text in REJECT_EXAMPLES:
            texts.append(text)
            self.labels.append(REJECT)

        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        LOG.info("Voice command matcher ready.")

    def rank(self, text):
        tokens = self.model.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        if len(tokens) > self.model.max_seq_length:
            return []

        query = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Dot product equals cosine similarity for normalized embeddings.
        similarities = self.embeddings @ query
        if not self.np.isfinite(similarities).all():
            raise RuntimeError("Command matcher returned invalid scores.")

        # Compare distinct classes, not two examples of the same class.
        best_by_class = {}
        for label, similarity in zip(self.labels, similarities):
            score = float(similarity)
            best_by_class[label] = max(
                score, best_by_class.get(label, -1.0),
            )

        return sorted(
            best_by_class.items(),
            key=lambda item: item[1],
            reverse=True,
        )


def warmup_command_matcher():
    """Call inside the existing speech worker during startup."""
    global _MATCHER
    with _MODEL_LOCK:
        if _MATCHER is None:
            _MATCHER = SemanticMatcher()


def _rank(text):
    global _MATCHER
    with _MODEL_LOCK:
        if _MATCHER is None:
            _MATCHER = SemanticMatcher()
        return _MATCHER.rank(text)


def interpret_command(transcript):
    # text = normalize(transcript)
    text = strip_politeness(normalize(transcript))

    if not text:
        return response(
            transcript, "no_speech", "No speech was recognized.",
        )

    if len(text) > 400:
        return clarify(
            transcript, "Please use a shorter request naming one target.",
        )

    if COMPLEX_COMMAND.search(text):
        return clarify(
            transcript,
            "Please repeat only the target you want. "
            "I cannot reliably interpret this correction, "
            "negation, or combined request.",
        )

    mentions = list(MENTION.finditer(text))
    mentioned_targets = {
        ALIASES[match.group(0)] for match in mentions
    }

    if len(mentioned_targets) > 1:
        return clarify(
            transcript,
            "More than one object was mentioned. Which one should I select?",
            candidates=sorted(mentioned_targets),
        )

    # Extract only explicit horizontal phrases.
    sides = {
        match.group("side") for match in SIDE_PHRASE.finditer(text)
    }
    sides = {
        "center" if side in ("centre", "middle") else side
        for side in sides
    }

    if len(sides) > 1:
        return clarify(
            transcript, "Please choose one side: left, center, or right.",
        )

    horizontal = next(iter(sides), None)
    query = " ".join(SIDE_PHRASE.sub(" ", text).split())

    # Mask whole known object names before checking unsupported modifiers.
    remainder = MENTION.sub(" ", query)
    if UNSUPPORTED_QUALIFIER.search(remainder):
        return clarify(
            transcript,
            "That description needs more than object-class matching. "
            "Please name the object, optionally with 'on my left' "
            "or 'on my right'.",
        )

    direct = DIRECT_COMMAND.fullmatch(query)
    if direct:
        return response(
            transcript,
            "resolved",
            "Target request understood.",
            target=ALIASES[direct.group("object")],
            horizontal=horizontal,
            method="explicit",
        )

    # An implicit seating request does not uniquely mean "chair".
    if not mentioned_targets and GENERIC_SEATING.search(query):
        choices = [
            name for name in ("chair", "sofa", "bench")
            if name in SUPPORTED
        ]
        return clarify(
            transcript,
            "Please name the seating object you want.",
            candidates=choices,
        )

    ranked = _rank(query)
    if not ranked:
        return clarify(
            transcript, "Please use a shorter request naming one target.",
        )

    winner, score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = score - runner_up_score

    diagnostics = {
        "method": "semantic",
        "similarity": round(score, 4),
        "margin": round(margin, 4),
        "candidates": [
            {"target": label, "similarity": round(value, 4)}
            for label, value in ranked
            if label != REJECT
        ][:3],
    }

    if winner == REJECT:
        return response(
            transcript,
            "unsupported_target",
            "I could not identify a supported target request. "
            "Please name an object from the target list.",
            **diagnostics,
        )

    # Never replace an explicitly mentioned object with another class.
    if mentioned_targets and winner not in mentioned_targets:
        return clarify(
            transcript,
            "I recognized an object name, but the request is unclear. "
            "Please say which object you want to select.",
            **diagnostics,
        )

    if score < MIN_SIMILARITY or margin < MIN_MARGIN:
        return clarify(
            transcript,
            "I am not certain which object you mean. "
            "Please name the target more specifically.",
            **diagnostics,
        )

    return response(
        transcript,
        "resolved",
        "Target request understood.",
        target=winner,
        horizontal=horizontal,
        **diagnostics,
    )