"""Per-interaction logging — the substrate for the memory/logs UI and eval.

In healthcare, logs are a PHI vector: traces get shipped, indexed, retained. So we
log DECISION metadata (the plan, which tools ran, success/failure, latency, route) —
and the message is PHI-REDACTED before it's written. Record content (a chart, a
clinical note) is never logged at all; only ids/metadata. This is best-effort masking
at the log boundary, not a DLP pipeline (SECURITY.md names the gap).
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
TRACE_PATH = os.path.join(LOG_DIR, "traces.jsonl")

# PII/PHI carriers. Patient ids (P####) and MRN-like tokens are masked too — in this
# domain an id ties a log line to a person's chart.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_LONG_NUM_RE = re.compile(r"\b\d{7,}\b")           # account/MRN-length numbers
_DOB_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")      # ISO dates of birth
_PID_RE = re.compile(r"\bP\d{2,}\b", re.IGNORECASE)  # internal patient ids (any case)


def redact(text: str) -> str:
    """Best-effort masking of emails, SSNs, phones, DOBs, long numbers, patient ids."""
    if not text:
        return text
    text = _EMAIL_RE.sub("[email]", text)
    text = _SSN_RE.sub("[ssn]", text)
    text = _PHONE_RE.sub("[phone]", text)
    text = _DOB_RE.sub("[date]", text)
    text = _PID_RE.sub("[patient-id]", text)
    text = _LONG_NUM_RE.sub("[redacted-number]", text)
    return text


def record(*, message, route, plan=None, tools=None, success=None, escalated=None,
           grounded=None, latency_ms=None, fell_back=None, advice_blocked=None) -> dict:
    """Append one interaction trace (message redacted, no record content). Returns the
    record so callers can display it. Never raises into the request path."""
    rec = {
        "trace_id": uuid.uuid4().hex[:8],
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": redact(message),
        "route": route,                 # emergency | agent | deferral
        "plan": plan,                   # list of tool steps (the goal decomposition)
        "tools": tools,                 # [{tool, status}]
        "success": success,
        "escalated": escalated,
        "grounded": grounded,
        "fell_back": fell_back,         # LLM phrasing rejected -> deterministic shipped
        "advice_blocked": advice_blocked,  # no-advice floor fired -> deferral
        "latency_ms": latency_ms,
    }
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return rec


def read_recent(limit: int = 50) -> list[dict]:
    """Most recent traces (newest first); one bad line never zeroes the history."""
    if not os.path.exists(TRACE_PATH):
        return []
    try:
        with open(TRACE_PATH, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return list(reversed(out))[:limit]


def metrics(limit: int = 1000) -> dict:
    traces = read_recent(limit)
    total = len(traces)
    by_route, tool_calls, tool_fail = {}, 0, 0
    for t in traces:
        by_route[t.get("route")] = by_route.get(t.get("route"), 0) + 1
        for tc in (t.get("tools") or []):
            tool_calls += 1
            # 'denied' is a CORRECT default-deny outcome, not a tool failure.
            if tc.get("status") not in ("ok", "success", "empty", "denied"):
                tool_fail += 1
    escalated = sum(1 for t in traces if t.get("escalated"))
    return {
        "total": total,
        "by_route": by_route,
        "escalated": escalated,
        "tool_calls": tool_calls,
        "tool_success_rate": (1 - tool_fail / tool_calls) if tool_calls else 1.0,
    }


def clear() -> None:
    if os.path.exists(TRACE_PATH):
        os.remove(TRACE_PATH)
