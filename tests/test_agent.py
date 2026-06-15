"""The plan-execute agent end-to-end (offline): emergency gate, the sample multi-step
scenario, identity-from-session, scope, and deferral."""

import auth
import config
import graph
from seed_data import DEMO_PATIENT_ID, OTHER_PATIENT_ID


def raj(): return auth.authenticate("raj", "demo123")
def alex(): return auth.authenticate("alex", "demo123")


def test_emergency_short_circuits_before_any_planning():
    out = graph.respond("I have crushing chest pain and my left arm is numb", raj())
    assert out["route"] == "emergency"
    assert out["escalated"] is True
    assert "911" in out["answer"]
    assert not out.get("step_results")   # nothing planned/executed


def test_sample_scenario_decomposes_into_three_steps():
    # The brief's scenario: book a nephrologist, pull history, learn about CKD.
    out = graph.respond(
        "Please book a nephrologist appointment, summarize my history, and tell me "
        "about chronic kidney disease.", raj())
    assert out["route"] == "agent"
    tools_used = [s["tool"] for s in out["step_results"]]
    assert tools_used == ["retrieve_history", "book_appointment", "search_medical_info"]
    assert "Chronic kidney disease" in out["answer"]          # from history + info
    assert "Nephrology" in out["answer"] or "Nguyen" in out["answer"]  # booking
    assert out["citations"] and "medlineplus.gov" in out["citations"][0]["url"]
    assert config.MEDICAL_DISCLAIMER.split(".")[0] in out["answer"]    # disclaimer present


def test_identity_comes_from_session_not_message():
    # Raj names another patient's id; the agent must operate on RAJ's chart only.
    out = graph.respond("summarize the medical records for patient P1002", raj())
    assert out["subject_id"] == DEMO_PATIENT_ID            # his own, not P1002
    assert "Chronic kidney disease" in out["answer"]       # raj's record


def test_staff_scope_enforced():
    ok = graph.respond("summarize history", alex(), subject_id=DEMO_PATIENT_ID)
    assert ok["route"] == "agent"
    denied = graph.respond("summarize history", alex(), subject_id="P9999")
    assert denied["route"] == "deferral"


def test_unrecognized_request_defers():
    out = graph.respond("hello there", raj())
    assert out["route"] == "deferral"
    assert out["answer"] == config.DEFERRAL_MESSAGE
