"""Long-term patient memory — persisted across sessions, scoped, redacted.

Two guards against the classic memory failure (leaking one patient's PHI into
another's prompt):
  1. Memory is keyed by patient_id and only ever loaded for the can_access'd subject —
     load_memory takes the subject and reads only that patient's rows.
  2. Content is PHI-redacted before it's stored, so the memory file itself carries the
     minimum (a defense-in-depth layer; see SECURITY.md for the production gap).
"""

import db
import observability


def load_memory(subject_id: str, limit: int = 5) -> list[str]:
    """Recent long-term context for ONE patient (most recent first)."""
    if not subject_id:
        return []
    return [r["content"] for r in db.list_memory(subject_id, limit)]


def save_memory(subject_id: str, content: str) -> bool:
    """Persist a short context note for a patient, redacted before storage."""
    if not subject_id or not content:
        return False
    return db.add_memory(subject_id, observability.redact(content))
