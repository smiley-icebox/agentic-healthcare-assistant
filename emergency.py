"""Emergency red-flag gate — the highest-liability control, deliberately the FIRST
thing that runs, on the RAW message, before any LLM/planner sees it.

If a message looks like an emergency, the agent is short-circuited entirely and a
FIXED, non-LLM message (config.EMERGENCY_MESSAGE: call 911 / 988) is returned. The
model never gets the chance to "handle" an emergency by booking a routine slot.

Tuned for HIGH RECALL — over-firing (a precaution to call 911) is safe; under-firing
is catastrophic. A code gate is the floor; an LLM classifier could add defense-in-depth
but never replaces this.
"""

import re

# Symptom/intent phrases that must trigger immediate escalation.
_RED_FLAGS = [
    r"chest pain", r"heart attack",
    r"can'?t breathe", r"cannot breathe", r"can'?t breath", r"short(ness)? of breath",
    r"struggling to breathe", r"trouble breathing",
    r"kill myself", r"suicid", r"end my life", r"want to die", r"harm myself", r"hurt myself",
    r"overdose", r"overdosed", r"took too many",
    r"stroke", r"face (is )?droop", r"slurred speech",
    r"sudden (weakness|numbness)", r"numb(ness)? (in|on) (one|my) (side|arm)",
    r"severe bleeding", r"bleeding (a lot|heavily|badly)", r"won'?t stop bleeding",
    r"anaphyla", r"throat (is )?closing", r"can'?t swallow", r"trouble swallowing",
    r"unconscious", r"unresponsive", r"passed out", r"collapsed",
    r"seizure", r"choking",
]
_RED_FLAG_RE = re.compile("|".join(_RED_FLAGS), re.IGNORECASE)


def is_emergency(message: str) -> bool:
    """True if the raw message contains an emergency red flag."""
    return bool(_RED_FLAG_RE.search(message or ""))
