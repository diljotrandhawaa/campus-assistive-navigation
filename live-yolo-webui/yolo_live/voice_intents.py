"""Exact control intents, followed by the existing semantic target matcher.

No session state lives here: each browser owns its clarification context.
Do not execute commands from OCR or from spoken output.
"""

import re

from .voice_commands import (
    interpret_command as interpret_target, normalize, strip_politeness,
)


CONTROL_PHRASES = {
    "start_camera": ("start", "start camera", "start the camera", "turn on the camera"),
    "stop_camera": ("stop camera", "stop the camera", "turn off the camera"),
    "stop_all": ("stop", "stop everything", "stop all", "shut down"),
    "cancel_target": (
        "cancel", "cancel target", "cancel the target", "cancel search",
        "cancel the search", "stop searching", "forget the target", "clear target",
    ),
    "pause": ("pause", "pause guidance", "pause the guidance", "be quiet", "mute"),
    "resume": ("resume", "resume guidance", "continue", "unmute"),
    "repeat": ("repeat", "repeat that", "say that again", "repeat the instruction"),
    "status": (
        "status", "what is happening", "whats happening", "what are you doing",
        "what are you looking for", "where is my target",
    ),
    "help": ("help", "help me", "what can i say", "what commands can i use"),
    "targets": ("list targets", "what can you find", "what objects can you find"),
    "read_text": (
        "read", "read text", "read the text", "read the sign", "read this",
        "read that", "what does it say", "read the room number",
    ),
    "ocr_on": ("enable ocr", "turn on text reading", "enable text reading"),
    "ocr_off": ("disable ocr", "turn off text reading", "disable text reading"),
    "handsfree_on": ("hands free on", "handsfree on", "keep listening"),
    "handsfree_off": ("hands free off", "handsfree off", "stop listening"),
}
CONTROLS = {phrase: action for action, phrases in CONTROL_PHRASES.items() for phrase in phrases}
SIDE = re.compile(
    r"(?:(?:the|that) )?(?:one )?(?:(?:on|to) )?(?:(?:the|my) )?"
    r"(?P<side>left|right|center|centre|middle)(?: one| side)?"
)


def interpret_command(transcript):
    text = strip_politeness(normalize(transcript))
    text = re.sub(r"^(?:can|could|would|will) you (?:please )?", "", text)
    text = strip_politeness(text)
    action = CONTROLS.get(text)
    if action:
        return {
            "status": "resolved", "transcript": transcript, "intent": action,
            "target": None, "horizontal": None, "method": "control",
            "message": "Command understood.",
        }
    side = SIDE.fullmatch(text)
    if side:
        horizontal = side.group("side")
        return {
            "status": "resolved", "transcript": transcript, "intent": "clarify_side",
            "target": None,
            "horizontal": "center" if horizontal in ("centre", "middle") else horizontal,
            "method": "context", "message": "Side understood.",
        }
    choice = re.fullmatch(r"(?:the )?(first|second|third)(?: one)?", text)
    if choice:
        return {
            "status": "resolved", "transcript": transcript, "intent": "clarify_choice",
            "choice": ("first", "second", "third").index(choice.group(1)),
            "target": None, "horizontal": None, "method": "context",
            "message": "Choice understood.",
        }
    result = interpret_target(text)
    # Keep the user's original wording in the UI and diagnostic response.
    result["transcript"] = transcript
    if result["status"] == "resolved":
        result["intent"] = "select_target"
    return result
