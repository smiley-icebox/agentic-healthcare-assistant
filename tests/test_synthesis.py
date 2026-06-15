"""The LLM synthesis path (graph._llm_answer) — exercised with a STUB chat model so the
safety gates that only run when USE_LLM is on are actually tested, with no API key.

These cover the gaps the production review flagged: advisory model output, ungrounded
model output, empty output, and — the project's headline guarantee — that a dropped
allergy/alert is caught and the answer falls back to the deterministic one."""

import auth
import config
import graph
import llm
from seed_data import DEMO_PATIENT_ID


class _FakeMsg:
    def __init__(self, text): self.content = text


class _FakeModel:
    """Stands in for ChatAnthropic: .invoke(messages) returns canned text."""
    def __init__(self, text): self._text = text
    def invoke(self, _messages): return _FakeMsg(self._text)


def _use_llm_with(monkeypatch, text):
    """Turn USE_LLM on and make every llm.chat_model() return our canned reply."""
    monkeypatch.setattr(config, "USE_LLM", True)
    monkeypatch.setattr(llm, "chat_model", lambda *a, **k: _FakeModel(text))


def _raj():
    return auth.authenticate("raj", "demo123")


HISTORY_REQ = "Summarize my medical history."
INFO_REQ = "Tell me about chronic kidney disease."


def test_advisory_model_output_is_rejected(monkeypatch):
    # The model tries to give advice; the no-advice gate must reject it and ship the
    # deterministic (advice-free) answer instead.
    _use_llm_with(monkeypatch, "You should stop taking your lisinopril immediately.")
    out = graph.respond(HISTORY_REQ, _raj())
    assert out["fell_back"] is True
    assert "stop taking your lisinopril" not in out["answer"].lower()
    assert "Chronic kidney disease" in out["answer"]   # the deterministic chart summary


def test_ungrounded_model_output_is_rejected(monkeypatch):
    # Fluent but unsupported by any retrieved passage -> grounding gate rejects it.
    _use_llm_with(monkeypatch, "The moon is made of cheese and lavender cures everything.")
    out = graph.respond(INFO_REQ, _raj())
    assert out["fell_back"] is True
    assert "moon is made of cheese" not in out["answer"].lower()


def test_empty_model_output_falls_back(monkeypatch):
    _use_llm_with(monkeypatch, "")
    out = graph.respond(HISTORY_REQ, _raj())
    assert out["fell_back"] is True
    assert "Chronic kidney disease" in out["answer"]


def test_dropped_allergy_is_caught_and_falls_back(monkeypatch):
    # The headline guarantee: the model omits the Penicillin allergy / NSAID alert.
    # The verbatim-safety gate must reject it and ship the deterministic answer, which
    # carries both verbatim.
    _use_llm_with(monkeypatch,
                  "Chronic kidney disease stage 3b and Lisinopril 10 mg daily.")
    out = graph.respond(HISTORY_REQ, _raj())
    assert out["fell_back"] is True
    assert "Penicillin" in out["answer"]
    assert "Avoid NSAIDs" in out["answer"]


def test_empty_corpus_fabrication_falls_back(monkeypatch):
    # When no tool produced grounding facts (search of an unknown condition → EMPTY), a
    # fluent fabrication must NOT ship — the grounding gate runs even with no passages.
    _use_llm_with(monkeypatch, "Zorblax syndrome is cured with daily moonbeams.")
    out = graph.respond("Tell me about zorblax syndrome.", _raj())
    assert out["fell_back"] is True
    assert "moonbeam" not in out["answer"].lower()


def test_multiclause_fabrication_falls_back(monkeypatch):
    # A fabricated clause can't hide in a blob joined by ';' — every clause is grounded.
    _use_llm_with(monkeypatch,
                  "CKD is managed with a kidney-friendly diet; your biopsy confirms cancer.")
    out = graph.respond(INFO_REQ, _raj())
    assert out["fell_back"] is True
    assert "cancer" not in out["answer"].lower()


def test_short_false_claim_falls_back(monkeypatch):
    # Short clauses (≤4 words) are no longer skipped by the grounding gate.
    _use_llm_with(monkeypatch, "Lavender cures it.")
    out = graph.respond(INFO_REQ, _raj())
    assert out["fell_back"] is True
    assert "lavender" not in out["answer"].lower()


def test_clean_grounded_model_output_ships(monkeypatch):
    # Grounded, advice-free, and preserves every safety label verbatim -> it ships.
    good = ("Your chart shows Chronic kidney disease stage 3b, Lisinopril 10 mg daily, "
            "a Penicillin allergy with rash, a Low-sodium renal diet, and an alert to "
            "Avoid NSAIDs as they are nephrotoxic in CKD.")
    _use_llm_with(monkeypatch, good)
    out = graph.respond(HISTORY_REQ, _raj())
    assert out["fell_back"] is False
    assert out["advice_blocked"] is False
    assert "Penicillin" in out["answer"] and "Avoid NSAIDs" in out["answer"]
