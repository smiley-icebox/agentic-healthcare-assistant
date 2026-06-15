# Agentic Healthcare Assistant — Project Writeup

Agentic Healthcare Assistant for Medical Task Automation
Applied GenAI Capstone · STAR format

---

## Situation

Patients interacting with a clinic touch several disjoint systems to do one thing.
"Book me a kidney specialist, remind me what's on my chart, and what is chronic kidney
disease?" is a single human request that spans an appointments system, an electronic
record, and a medical-information source — and it carries real risk: the wrong patient's
chart, a missed emergency, a paraphrased allergy, or a model confidently giving medical
advice are all direct harms, not bad UX.

## Task

Build an **agentic** assistant that plans and automates multi-step medical tasks:
goal decomposition, tool use (appointments, records, knowledge retrieval), RAG over
trusted sources, and memory — plus the LLMOps layer (evaluation, a Streamlit dashboard,
logging). Then take it past the brief toward something safe enough to actually run.

## Action

### The central design decision: an agent, fenced in by deterministic gates

Unlike a 3-way classifier, this request genuinely needs an **agent** — the steps and
their order aren't known until the message is read. So I built a **plan-and-execute
agent** on LangGraph (`planner → executor → synthesizer`). But the LLM is confined to
the two things only it can do — **decompose** fuzzy language into steps and **phrase** a
warm reply — and is fenced out of everything else:

- **Code owns control flow and facts.** The emergency check, who-can-access-what, slot
  booking, chart contents, citations, and the no-advice check are all deterministic.
- **The plan is untrusted output.** `validate_plan` drops any non-allowlisted tool or
  arg before execution; `patient_id` is **never** an allowed arg.
- **Identity comes from the session, not the message.** Resolved in `guard`, so a prompt
  naming another patient's id can't move whose chart is touched.

The agent is **sandwiched between gates**: an emergency red-flag check runs *first* on
the raw message (before any LLM), and a no-advice validator runs *last* on the final
answer (whichever path produced it).

### Core build (the brief)

- **Goal decomposition** — `planner.py` turns a request into an ordered tool sequence
  (LLM when a key is present; a keyword heuristic offline). Always allowlist-validated.
- **Four tools** — `retrieve_history`, `book_appointment`, `search_medical_info`,
  `manage_records`, each returning a uniform `{status, summary, data, citations}`.
- **RAG** — offline-first MedlinePlus/WHO retrieval indexed in **FAISS** over TF-IDF
  vectors (no torch), with code-attached citations and "as of" staleness.
- **Memory** — per-patient long-term notes, scoped and PHI-redacted before storage.
- **Streamlit** dashboard, **PHI-redacted logging**, and **evaluation**.

### Safety hardening (the part that matters in healthcare)

- **Emergency gate (`emergency.py`)** — deterministic, first, high-recall; returns a
  fixed 911/988 message with no LLM in the path. Fires in ~1 ms vs ~10 s for the agent.
- **Default-deny access (`auth.can_access`)** — patient → own chart; staff → an explicit
  assigned list. Enforced at both the graph and the tool layer (belt and suspenders).
- **No-advice floor (`safety.py`)** — runs on the final answer regardless of path, so
  the deterministic fallback can't bypass it. Live source prose is second-person and
  reads as advice when quoted, so advisory live passages are rejected at retrieval in
  favor of the curated advice-free corpus.
- **Grounding** — a deterministic token-overlap check that every substantive sentence is
  supported by a tool result (medical passages + the patient's structured chart + the
  tools' own summaries). Unsupported LLM phrasing falls back to the grounded-by-
  construction deterministic answer; a dropped allergy/alert is caught the same way.
- **No double-booking** — slot booking is a single conditional `UPDATE ... WHERE
  status='available'` asserting `rowcount == 1`, proven race-safe with two real threads.
- **Append-only records + audit** — every write lands an immutable audit row in the same
  transaction; allergies/alerts are surfaced **verbatim**, never LLM-summarized.

### Transactional storage seam

All DB access sits behind a repository with versioned migrations; SQLite today, Postgres
a drop-in at the factory. Slots use `UNIQUE(doctor_id, start_at)` as the unit of booking
concurrency, and the data layer never raises into the request path.

## Result

- All required capabilities work end-to-end. On the **versioned eval set**, the
  deterministic gates hold: emergency recall + precision, deferral accuracy, identity
  integrity, no-advice rate, citation validity, and tool success all **1.0**, with plan
  **recall 1.0** (the planner never drops a needed step). With a key, LangChain
  **`QAEvalChain` answer correctness is 1.0** and an **independent** tone judge (a
  different model, to blunt self-grading bias) scores empathy/clarity.
- **Honest about the metrics:** offline, the `no_advice_rate` and `groundedness`
  numbers are grounded-by-construction (the deterministic answer can't violate them), so
  they're 1.0 by design. The load-bearing proof that those *gates* actually catch bad
  LLM output — advisory phrasing, an ungrounded claim, a **dropped allergy** — lives in
  `test_synthesis.py`, which runs the LLM path against a stub model. The eval also
  reports gate-activation telemetry (`advice_blocked`, `fell_back`) so a regression is
  visible rather than masked.
- **37 automated tests pass with no API call** — the LLM, live search, and eval graders
  all sit behind flags `conftest` forces off, so booking concurrency, access control,
  the emergency gate, the grounding / verbatim-safety / no-advice gates, and the eval
  parsers are all verifiable offline.
- The dashboard makes the agent legible: per message you see the **goal decomposition
  (the plan)**, each tool's status, the grounded answer with citations, plus the
  appointment book, the chart (allergies/alerts flagged), a PHI-redacted decision log,
  and the eval scorecard.

## Engineering decisions worth calling out

| Decision | Why |
|----------|-----|
| Agent (plan-execute), not a static workflow | The steps/order aren't known until the message is read — this is the case that warrants an agent. |
| …but the plan is validated outside the LLM | A manipulated plan can't widen access or invent a tool; `patient_id` is never an arg. |
| Identity from the session, never the message | The one control that stops cross-patient PHI access (IDOR). |
| Emergency gate first, deterministic, no LLM | A model must never get the chance to "handle" an emergency by booking a slot. |
| No-advice validator on the final answer, both paths | A floor the deterministic fallback can't bypass; the disclaimer alone is not a control. |
| Citations attached in code, from an allowlist | The LLM can't fabricate a source; provenance is verifiable. |
| Per-step failure state machine | An empty-but-valid result (`empty`) is never confused with a tool error; failures are surfaced, not papered over. |
| Eval split: deterministic gates vs. LLM judge | Safety/correctness are checked deterministically; the judge is scoped to tone only, on an independent model. |
| Reproducible eval inputs | The eval pins live retrieval OFF so answers are graded over a fixed corpus, not MedlinePlus content that changes daily. Live search stays a runtime feature. |

**A deliberate safety/fluency tradeoff.** When an LLM phrasing fails any gate (grounding,
verbatim safety, no-advice), the system ships the deterministic answer instead. That
answer is complete and correct but more clinical, so the independent tone judge scores
its empathy lower — a tradeoff I take on purpose for a healthcare assistant, and one the
eval now reports honestly via `fell_back` / `advice_blocked` telemetry rather than hiding.

## What production would still demand (documented, not built)

The seams are in place; these are the real implementations behind them:

- A real **identity provider + authorization service** behind `auth.py` (demo seeds
  users in memory).
- The actual **Postgres** repository (interface + portable SQL ready), with **PHI
  encryption at rest** and a real **DLP** pipeline at the storage boundary (redaction is
  log-boundary only today).
- **Embeddings + a managed vector store** behind the RAG retriever (TF-IDF + FAISS over a
  small corpus today).
- An **LLM classifier as defense-in-depth** on top of the regex emergency/no-advice
  floors (never as a replacement), plus a clinician-in-the-loop review queue.
- A **right-sized model tier** and cost/latency budget; HIPAA posture (BAAs, access-log
  retention, breach procedures) — all out of scope for a coursework demo.
