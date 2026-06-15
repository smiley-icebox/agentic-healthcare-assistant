"""Post-synthesis no-advice validator — the control the disclaimer can't be.

A patient reads the answer, not the footer. And medical *advice* emerges from
combining grounded parts ("your record shows hypertension" + "untreated hypertension
causes stroke" → "you should..."). So after the synthesizer writes an answer, this
deterministic validator scans for advisory/imperative clinical language directed at
the user. If it fires, the answer is REPLACED with a deferral — never shipped.

This is the analogue of the prior project's reply guard, but for the info/advice line
specific to healthcare. It's a floor, not a substitute for the synthesizer being
prompted to stay informational.
"""

import re

# Imperative / advisory clinical phrasings aimed at the user. Kept specific to avoid
# nuking legitimate informational text ("people with CKD should avoid NSAIDs" is
# general info; "you should stop taking your lisinopril" is advice).
_ADVICE_PATTERNS = [
    r"\byou should\b", r"\byou need to\b", r"\byou must\b", r"\byou ought to\b",
    r"\bi (recommend|suggest|advise)\b", r"\bi'?d (recommend|suggest|advise)\b",
    r"\bi would (recommend|suggest|advise)\b",
    r"\byou (can|could) safely\b", r"\bit'?s safe (for you )?to\b",
    r"\b(start|stop|take|increase|decrease|switch|change|double|halt|discontinue)\s+"
    r"(your|taking|the)\b",
    r"\bstop taking\b", r"\bdouble your dose\b", r"\bup your dose\b",
    r"\byour diagnosis is\b",
    # SPECULATIVE diagnosis aimed at the user (the assistant guessing a condition). A
    # plain restatement of a CHARTED diagnosis ("you have a recorded diagnosis of CKD")
    # is legitimate and is left to the grounding gate — only speculation is advice here.
    r"\byou (likely|probably|may|might|could|seem to|appear to) have\b.*\b(disease|cancer|condition|diabetes)\b",
    r"\byou don'?t need\b", r"\bno need to see\b",
    # Passive / impersonal advice — the same act, just depersonalized.
    r"\bit('?s| is)? (recommended|advisable|advised|best|important) (that you |to )\b",
    r"\b(stopping|starting|increasing|decreasing|discontinuing|halting) (the|your)\b",
    r"\bthe (best|recommended|appropriate|right) (step|course|option|thing) (is|would be) to\b",
    r"\b(should|must|need to) be (started|stopped|increased|decreased|discontinued)\b",
    # Imperative LIFESTYLE directives aimed at the patient — common in raw source prose
    # ("Don't smoke. Lose weight. Control your blood pressure."). Informational third-
    # person phrasing ("managed with a low-salt diet", "smoking cessation helps") has no
    # second-person/imperative directive and is NOT caught. This is what keeps advisory
    # LIVE source content out — it's rejected at retrieval in favor of the curated corpus.
    r"\b(don'?t|do not|never|always) (smoke|drink|eat|use|skip|stop|take|wait|delay)\b",
    r"\blose weight\b", r"\bbe (physically )?active\b", r"\bquit smoking\b",
    r"\b(control|keep|limit|monitor|check|maintain|lower|raise|reduce|increase|watch|track|manage) your\b",
    r"\bchoose (foods|healthy|whole|low-fat|fruits)\b", r"\blimit the amount\b",
    r"\byou (drink|smoke|eat)\b",
]
_ADVICE_RE = re.compile("|".join(_ADVICE_PATTERNS), re.IGNORECASE)


def contains_advice(text: str) -> bool:
    """True if the text gives (or reads as) personalized medical advice to the user."""
    return bool(_ADVICE_RE.search(text or ""))
