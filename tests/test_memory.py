"""Long-term patient memory — PHI-redacted storage and FAISS-backed semantic retrieval,
scoped to one patient (no cross-patient leakage)."""

import memory
from seed_data import DEMO_PATIENT_ID, OTHER_PATIENT_ID


def test_search_memory_is_relevant_and_scoped():
    memory.save_memory(DEMO_PATIENT_ID, "Chart summary — diagnosis: Chronic kidney disease, stage 3b")
    memory.save_memory(DEMO_PATIENT_ID, "Request: book a cardiology appointment")
    memory.save_memory(DEMO_PATIENT_ID, "Request: low-sodium renal diet guidance")
    memory.save_memory(OTHER_PATIENT_ID, "Request: maria asked about a flu shot")

    hits = memory.search_memory(DEMO_PATIENT_ID, "kidney disease history", k=2)
    assert hits, "expected FAISS retrieval to return something"
    assert any("kidney" in h.lower() for h in hits)          # relevance
    assert all("maria" not in h.lower() for h in hits)       # scoped to this patient only


def test_search_memory_redacts_before_store():
    memory.save_memory(DEMO_PATIENT_ID, "Contact patient at 555-123-4567 about P1002")
    hits = memory.search_memory(DEMO_PATIENT_ID, "contact patient phone", k=5)
    blob = " ".join(hits)
    assert "555-123-4567" not in blob and "P1002" not in blob   # PHI masked at the boundary


def test_search_memory_falls_back_when_too_little():
    # One note (or no query) → recency fallback, never a crash.
    memory.save_memory(OTHER_PATIENT_ID, "Request: single note")
    assert memory.search_memory(OTHER_PATIENT_ID, "", k=3)        # empty query → recency
    assert memory.search_memory(OTHER_PATIENT_ID, "anything", k=3)  # <2 notes → recency
