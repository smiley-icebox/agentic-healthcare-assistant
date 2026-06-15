"""The planner — decomposes a patient query into an ordered list of tool steps.

CRITICAL: the plan is treated as UNTRUSTED output. `validate_plan` drops any step
whose tool isn't in the allowlist and strips any arg that isn't permitted for that
tool — and `patient_id` is NEVER an allowed arg (identity comes from the session, not
the plan). So even a manipulated LLM plan can't widen access or invent a tool.

Two planners share the same validation:
  - the LLM planner (when USE_LLM) emits a JSON plan;
  - a deterministic keyword planner is the offline fallback (and what tests exercise).
"""

import json
import re

import config
import llm

# The tool allowlist + the args each tool may legitimately receive from a plan.
# patient_id is intentionally absent everywhere — it's injected from the session.
ALLOWED_ARGS = {
    "retrieve_history": set(),
    "book_appointment": {"specialty", "when"},
    "search_medical_info": {"condition"},
    "manage_records": {"record_type", "label", "value", "note"},
}
TOOL_NAMES = tuple(ALLOWED_ARGS)

_SPECIALTY_HINTS = {
    "nephrolog": "Nephrology", "kidney": "Nephrology", "renal": "Nephrology",
    "cardiolog": "Cardiology", "heart": "Cardiology",
    "general prac": "General Practice", "gp": "General Practice", "checkup": "General Practice",
}
_CONDITION_HINTS = {
    "kidney": "chronic kidney disease", "ckd": "chronic kidney disease", "renal": "chronic kidney disease",
    "blood pressure": "high blood pressure", "hypertension": "high blood pressure",
    "diabetes": "diabetes", "blood sugar": "diabetes",
    "penicillin": "penicillin allergy",
}


def validate_plan(steps) -> list[dict]:
    """Keep only allowlisted tools + permitted args. The security boundary."""
    out = []
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        tool = s.get("tool")
        if tool not in ALLOWED_ARGS:
            continue
        args = {k: v for k, v in (s.get("args") or {}).items() if k in ALLOWED_ARGS[tool]}
        out.append({"tool": tool, "args": args})
    return out


def _condition_from(message: str) -> str:
    m = (message or "").lower()
    for hint, cond in _CONDITION_HINTS.items():
        if hint in m:
            return cond
    mt = re.search(r"(?:about|for|options for|symptoms of|treat(?:ment)? for)\s+([a-z][a-z \-]{2,40})", m)
    return (mt.group(1).strip() if mt else (message or "").strip())[:60]


def _specialty_from(message: str) -> str:
    """The recognized specialty, or "" if none — never a silent default. Defaulting an
    unrecognized specialty to General Practice would book the WRONG clinician; an empty
    specialty lets book_appointment return an honest ERROR/EMPTY instead."""
    m = (message or "").lower()
    for hint, spec in _SPECIALTY_HINTS.items():
        if hint in m:
            return spec
    return ""


def heuristic_plan(message: str) -> list[dict]:
    """Deterministic keyword planner (offline fallback). Detects multiple intents so
    a multi-step query ('book a nephrologist, pull my history, and tell me about CKD')
    decomposes into several ordered steps."""
    m = (message or "").lower()
    steps = []
    if any(w in m for w in ("history", "summar", "my record", "past diagnos", "my chart", "medical record")):
        steps.append({"tool": "retrieve_history", "args": {}})
    if any(w in m for w in ("book", "appointment", "schedule", "see a ", "see the ")):
        steps.append({"tool": "book_appointment", "args": {"specialty": _specialty_from(m)}})
    if any(w in m for w in ("what is", "what are", "information", "tell me about", "treatment option",
                            "symptoms", "learn about", "explain")):
        steps.append({"tool": "search_medical_info", "args": {"condition": _condition_from(message)}})
    return validate_plan(steps)   # empty plan => the graph defers


_PLANNER_SYSTEM = """You are the PLANNER for a healthcare assistant. Decompose the
patient's request into an ordered list of tool steps. Output ONLY JSON: a list of
{"tool": <name>, "args": {...}}.

Tools and their ONLY allowed args:
- "retrieve_history": {}  — summarize the patient's own history
- "book_appointment": {"specialty", "when"} — book an appointment
- "search_medical_info": {"condition"} — look up trusted disease information
- "manage_records": {"record_type","label","value","note"} — add a record (staff only)

Rules: never include a patient id (identity is handled by the system). Use only the
tools/args above. If the request needs nothing, output []. Order steps sensibly
(e.g. retrieve history before searching treatment options)."""


def make_plan(message: str) -> list[dict]:
    """LLM plan when enabled, else the heuristic. Always passed through validate_plan."""
    if not config.USE_LLM:
        return heuristic_plan(message)
    try:
        msg = llm.chat_model(512, temperature=0).invoke(
            [("system", _PLANNER_SYSTEM), ("human", message)])
        text = llm.extract_text(getattr(msg, "content", "")).strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        steps = json.loads(text)
        plan = validate_plan(steps)
        return plan or heuristic_plan(message)   # fall back if the LLM gave nothing usable
    except Exception:
        return heuristic_plan(message)
