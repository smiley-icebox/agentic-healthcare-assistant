"""The agent's tools — the deterministic half of the system.

Every tool follows the same contract:
  - signature is `tool(session, subject_id, **kwargs)` — the SUBJECT patient is passed
    by the caller from the trusted session, NEVER from LLM/planner output.
  - the FIRST thing every tool touching a chart does is `can_access(session, subject_id)`
    (default-deny) — belt-and-suspenders on top of the graph's scope check.
  - returns a uniform ToolResult dict {status, summary, data, citations} where status is
    ok | empty | error | denied. The executor's failure policy reads `status`, so an
    empty-but-valid result is never confused with a tool error.

Facts come from the DB; the LLM never invents them. These tools return STRUCTURED facts
only — any grounded LLM phrasing happens later, in the synthesizer (graph._llm_answer),
which is gated for grounding, no-advice, and verbatim-safety before anything ships.
"""

from auth import can_access

OK, EMPTY, ERROR, DENIED = "ok", "empty", "error", "denied"


def result(status, summary, data=None, citations=None) -> dict:
    return {"status": status, "summary": summary, "data": data or {}, "citations": citations or []}


def _actor(session) -> str:
    return f"{session.role}:{session.username}"


# Free-text specialty -> canonical doctor specialty. The LLM planner may emit
# "nephrologist" / "kidney doctor"; doctors are filed under "Nephrology". The TOOL
# (not the planner) owns this mapping, because matching a real specialty is a fact.
_SPECIALTY_STEMS = {
    "nephro": "Nephrology", "kidney": "Nephrology", "renal": "Nephrology",
    "cardio": "Cardiology", "heart": "Cardiology",
    "general prac": "General Practice", "gp": "General Practice",
    "family": "General Practice", "primary care": "General Practice",
}


def _resolve_specialty(requested: str) -> str:
    """Map a requested specialty onto a specialty that actually exists in the
    directory. Falls through to the raw value (-> an honest EMPTY) if nothing matches."""
    import db
    known = [d["specialty"] for d in db.list_doctors()]
    r = (requested or "").strip().lower()
    for k in known:                                  # exact (case-insensitive)
        if k.lower() == r:
            return k
    for stem, canon in _SPECIALTY_STEMS.items():     # synonym / stem
        if stem in r and canon in known:
            return canon
    for k in known:                                  # loose substring either way
        if r and (r in k.lower() or k.lower() in r):
            return k
    return requested


def book_appointment(session, subject_id, specialty, when=None, **_) -> dict:
    """Find an available slot for a specialty and book it transactionally for the
    subject patient. Tries successive slots if one is claimed by a concurrent booking
    (the race-loser), so a transient slot conflict never fails the booking outright."""
    import db
    if not can_access(session, subject_id):
        return result(DENIED, "You're not authorized to book for this patient.")
    if not specialty:
        return result(ERROR, "No specialty was specified for the appointment.")

    specialty = _resolve_specialty(specialty)        # canonicalize before matching
    slots = db.find_available_slots(specialty, limit=10)
    if when:  # soft preference: prefer a slot whose date/time contains the hint
        preferred = [s for s in slots if when.lower() in s["start_at"].lower()]
        slots = preferred + [s for s in slots if s not in preferred]
    if not slots:
        return result(EMPTY, f"No available {specialty} slots right now.")

    for slot in slots:
        appt = db.book_slot(slot["slot_id"], subject_id, specialty, actor=_actor(session))
        if appt:  # exactly one writer wins a given slot; if claimed, try the next
            return result(
                OK,
                f"Booked a {specialty} appointment with {slot['doctor_name']} "
                f"on {appt['start_at']}.",
                data=appt,
            )
    return result(ERROR, f"All {specialty} slots were just taken — please try again.")


# Record types surfaced VERBATIM (never LLM-summarized) because a paraphrase error
# is direct harm: a missed allergy or a softened safety alert.
_VERBATIM_TYPES = ("allergy", "alert")
_SUMMARY_ORDER = ("diagnosis", "medication", "treatment", "allergy", "alert", "note")


def manage_records(session, subject_id, record_type, label, value=None, note=None,
                   supersedes=None, **_) -> dict:
    """Add a clinical record. STAFF ONLY (attendant/clinician) — a patient cannot
    write to their own chart. Append-only + audited in the repository. `supersedes` (a
    prior record_id) marks a correction; it is NOT a planner-supplied arg (not in
    planner.ALLOWED_ARGS), so it can only come from a trusted clinical-correction flow."""
    import db
    if not session.is_staff:
        return result(DENIED, "Only an attendant or clinician can update records.")
    if not can_access(session, subject_id):
        return result(DENIED, "You're not authorized to update this patient's record.")
    if not record_type or not label:
        return result(ERROR, "A record needs at least a type and a label.")
    rid = db.add_record(subject_id, record_type, label, value, note,
                        recorded_by=_actor(session), supersedes=supersedes)
    if rid is None:
        return result(ERROR, "Couldn't save the record — please try again.")
    return result(OK, f"Added {record_type}: {label} to the record.", data={"record_id": rid})


def retrieve_history(session, subject_id, **_) -> dict:
    """Return the patient's history as STRUCTURED facts (the LLM later only phrases
    these — it never parses facts out of prose, and never invents them). Safety-
    critical items (allergies, alerts) are carried verbatim for the synthesizer to
    surface unchanged."""
    import db
    if not can_access(session, subject_id):
        return result(DENIED, "You're not authorized to view this patient's history.")
    recs = db.list_records(subject_id)
    if not recs:
        # empty-but-valid: the patient genuinely has no history (distinct from error).
        return result(EMPTY, "No medical history is on file for this patient.")

    grouped: dict[str, list] = {}
    for r in recs:
        grouped.setdefault(r["record_type"], []).append(r)
    # The safety-critical LABEL (the allergen / the alert instruction) must appear in any
    # answer verbatim — a softened or dropped "Penicillin" / "Avoid NSAIDs" is direct
    # harm. The synthesizer enforces this; the value stays in `records` for grounding.
    verbatim = [r["label"] for r in recs if r["record_type"] in _VERBATIM_TYPES]
    # Deterministic, grounded-by-construction summary (the safe baseline; the
    # synthesizer may re-phrase it but can introduce no new clinical facts).
    lines = []
    for rtype in _SUMMARY_ORDER:
        for r in grouped.get(rtype, []):
            v = f" — {r['value']}" if r.get("value") else ""
            lines.append(f"{rtype}: {r['label']}{v}")
    return result(
        OK,
        "; ".join(lines),
        data={"records": recs, "types": sorted(grouped), "verbatim_safety": verbatim},
    )


def search_medical_info(session, subject_id=None, condition=None, **_) -> dict:
    """Retrieve trusted disease info (MedlinePlus/WHO, offline fallback). No patient
    data, so no access check. Returns an extractive summary + code-attached citations;
    grounding for any LLM phrasing is enforced at synthesis."""
    import knowledge
    if not condition:
        return result(ERROR, "No condition was specified to look up.")
    res = knowledge.search(condition)
    passages = res.get("passages") or []
    if not passages:
        return result(EMPTY, f"I couldn't find trusted information on '{condition}'.")
    top = passages[0]
    summary = top["content"] + top.get("as_of_note", "")
    return result(OK, summary, data={"passages": passages, "live": res.get("live")},
                  citations=res.get("citations", []))
