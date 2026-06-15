"""Booking: the tool path, access control, and the headline correctness claim —
no double-booking under concurrency (proven with two real threads, not asserted)."""

from concurrent.futures import ThreadPoolExecutor

import auth
import db
import tools
from seed_data import DEMO_PATIENT_ID, OTHER_PATIENT_ID


def _raj():
    return auth.authenticate("raj", "demo123")


def test_book_appointment_succeeds():
    out = tools.book_appointment(_raj(), DEMO_PATIENT_ID, "Nephrology")
    assert out["status"] == "ok"
    assert out["data"]["specialty"] == "Nephrology" and out["data"]["status"] == "booked"


def test_book_denied_for_unauthorized_patient():
    # Raj cannot book for Maria's chart.
    out = tools.book_appointment(_raj(), OTHER_PATIENT_ID, "Nephrology")
    assert out["status"] == "denied"


def test_book_empty_when_no_slots_for_specialty():
    out = tools.book_appointment(_raj(), DEMO_PATIENT_ID, "Dermatology")
    assert out["status"] == "empty"


def test_book_resolves_fuzzy_specialty_from_llm_plan():
    # The LLM planner emits free text ("nephrologist", "kidney doctor"); the tool must
    # resolve it to the canonical "Nephrology" and still book. (Regression: the live
    # plan said 'nephrologist' and booking wrongly returned EMPTY.)
    for term in ("nephrologist", "kidney doctor", "NEPHROLOGY"):
        out = tools.book_appointment(_raj(), DEMO_PATIENT_ID, term)
        assert out["status"] == "ok", f"{term!r} -> {out['status']}"
        assert out["data"]["specialty"] == "Nephrology"


def test_clinician_maps_to_doctor_and_sees_their_calendar():
    # The doctor view: a clinician is tied to a directory doctor and can read that
    # doctor's own schedule (the seeded GP appointment is on Dr. Lee's calendar).
    sess = auth.authenticate("drlee", "demo123")
    assert sess.doctor_id == "D_gp"
    calendar = db.list_appointments(doctor_id="D_gp")
    assert calendar and calendar[0]["patient_id"] == OTHER_PATIENT_ID


def test_no_double_booking_under_concurrency():
    # Two threads race to claim the SAME slot. Exactly one may win.
    slot_id = db.find_available_slots("Nephrology")[0]["slot_id"]

    def claim(pid):
        return db.book_slot(slot_id, pid, "Nephrology", actor=f"patient:{pid}")

    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(claim, [DEMO_PATIENT_ID, OTHER_PATIENT_ID]))

    winners = [r for r in results if r is not None]
    assert len(winners) == 1, f"expected exactly one winner, got {len(winners)}"
    # and the slot is now booked to exactly one patient
    appts = db.list_appointments()
    booked_for_slot = [a for a in appts if a["slot_id"] == slot_id]
    assert len(booked_for_slot) == 1
