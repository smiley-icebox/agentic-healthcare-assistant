"""Demo data: patients, doctors, open slots, and clinical records.

Patient P1001 (Raj) is seeded to match the brief's sample scenario — a 50-year-old
with chronic kidney disease who wants a nephrologist appointment — so the canonical
demo reproduces the brief end-to-end.
"""

import db

DEMO_PATIENT_ID = "P1001"          # Raj — the brief's 50yo CKD scenario
OTHER_PATIENT_ID = "P1002"         # Maria — used to prove access scoping

PATIENTS = [
    (DEMO_PATIENT_ID, "Raj Patel", "1976-04-12"),
    (OTHER_PATIENT_ID, "Maria Gomez", "1990-09-03"),
]

DOCTORS = [
    ("D_neph", "Dr. Nguyen", "Nephrology"),
    ("D_card", "Dr. Okafor", "Cardiology"),
    ("D_gp", "Dr. Lee", "General Practice"),
]

# (doctor_id, start_at) — available slots.
SLOTS = [
    ("D_neph", "2026-06-22T09:00:00+00:00"),
    ("D_neph", "2026-06-22T10:30:00+00:00"),
    ("D_neph", "2026-06-23T14:00:00+00:00"),
    ("D_card", "2026-06-22T11:00:00+00:00"),
    ("D_gp", "2026-06-21T08:30:00+00:00"),
]

# Clinical records for the CKD patient. Typed rows: the LLM only PHRASES these on
# read; it never parses facts back out of prose. Allergies/alerts are verbatim.
RECORDS = [
    (DEMO_PATIENT_ID, "diagnosis", "Chronic kidney disease", "stage 3b", None, "clinician:drlee"),
    (DEMO_PATIENT_ID, "medication", "Lisinopril", "10 mg daily", None, "clinician:drlee"),
    (DEMO_PATIENT_ID, "allergy", "Penicillin", "rash", None, "clinician:drlee"),
    (DEMO_PATIENT_ID, "treatment", "Low-sodium renal diet", "ongoing", None, "clinician:drlee"),
    (DEMO_PATIENT_ID, "alert", "Avoid NSAIDs", "nephrotoxic in CKD", None, "clinician:drlee"),
]


def seed() -> None:
    db.reset_db()
    for pid, name, dob in PATIENTS:
        db.add_patient(pid, name, dob)
    for did, name, spec in DOCTORS:
        db.add_doctor(did, name, spec)
    for did, start in SLOTS:
        db.add_slot(did, start)
    for pid, rtype, label, value, note, by in RECORDS:
        db.add_record(pid, rtype, label, value, note, recorded_by=by)


if __name__ == "__main__":
    seed()
    print(f"Seeded {len(PATIENTS)} patients, {len(DOCTORS)} doctors, {len(SLOTS)} slots, "
          f"{len(RECORDS)} records into {db.get_patient(DEMO_PATIENT_ID) and 'healthcare.db'}")
