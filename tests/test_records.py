"""Records management (staff-only, scoped) + grounded history retrieval."""

import auth
import db
import tools
from seed_data import DEMO_PATIENT_ID, OTHER_PATIENT_ID


def s(u): return auth.authenticate(u, "demo123")


# --- manage_records (write) --------------------------------------------------
def test_clinician_can_add_record():
    out = tools.manage_records(s("drlee"), DEMO_PATIENT_ID, "diagnosis", "Hypertension")
    assert out["status"] == "ok" and out["data"]["record_id"]


def test_patient_cannot_write_their_own_record():
    out = tools.manage_records(s("raj"), DEMO_PATIENT_ID, "diagnosis", "Self-diagnosis")
    assert out["status"] == "denied"


def test_staff_cannot_write_for_unassigned_patient(monkeypatch):
    # Force an attendant whose assignment list excludes the patient -> denied.
    sess = s("alex")
    object.__setattr__(sess, "assigned_patients", ())  # frozen dataclass: bypass for test
    out = tools.manage_records(sess, DEMO_PATIENT_ID, "note", "x")
    assert out["status"] == "denied"


def test_record_correction_supersedes_old_on_read():
    # Append-only with corrections: a superseding record hides the old one on read,
    # but the raw row stays in the table (audit-preserving).
    sess = s("drlee")
    rid = tools.manage_records(sess, DEMO_PATIENT_ID, "diagnosis", "Provisional dx")["data"]["record_id"]
    tools.manage_records(sess, DEMO_PATIENT_ID, "diagnosis", "Corrected dx", supersedes=rid)
    labels = [r["label"] for r in db.list_records(DEMO_PATIENT_ID)]
    assert "Corrected dx" in labels
    assert "Provisional dx" not in labels   # superseded row excluded on read


# --- retrieve_history (read) -------------------------------------------------
def test_retrieve_history_is_grounded_and_surfaces_verbatim_safety():
    out = tools.retrieve_history(s("raj"), DEMO_PATIENT_ID)
    assert out["status"] == "ok"
    assert "Chronic kidney disease" in out["summary"]
    # allergy + alert carried verbatim for the synthesizer to surface unchanged
    joined = " | ".join(out["data"]["verbatim_safety"])
    assert "Penicillin" in joined and "Avoid NSAIDs" in joined


def test_retrieve_history_denied_cross_patient():
    out = tools.retrieve_history(s("raj"), OTHER_PATIENT_ID)
    assert out["status"] == "denied"


def test_retrieve_history_empty_is_distinct_from_error():
    # drlee is assigned P1002 (Maria), who has no records -> empty, not error.
    out = tools.retrieve_history(s("drlee"), OTHER_PATIENT_ID)
    assert out["status"] == "empty"
