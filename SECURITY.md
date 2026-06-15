# Security Policy

The Agentic Healthcare Assistant is an **educational / portfolio project** (an Applied
GenAI capstone). It is **not a production system, not a medical device, and not a
source of medical advice.** It processes only synthetic demo data — there are no real
patients, charts, or secrets in this repository.

## Reporting a vulnerability

If you find a security issue you'd still like to report, please use **GitHub's private
vulnerability reporting** (the repository's **Security** tab → *Report a vulnerability*)
rather than opening a public issue. I'll respond as time allows — this is a personal
project, not a maintained service.

## What *is* handled (the safety-critical controls)

These are the load-bearing controls, each backed by tests:

- **Identity comes from the session, never the message.** Whose chart is read/written
  is fixed by the authenticated `Session`; a prompt that names another patient id
  (`P1002`) cannot move the subject. The planner is forbidden from ever carrying a
  `patient_id` arg (`planner.ALLOWED_ARGS`). (Test: `test_identity_comes_from_session`.)
- **Default-deny access (`auth.can_access`).** A patient sees only their own chart; an
  attendant/clinician only the patients on their explicit assignment list — a *checked*
  allowlist, never an implicit "staff sees everyone." Every chart read/write passes
  through it, at both the graph and the tool layer.
- **Emergency gate runs first, deterministically, before any LLM.** A red-flag message
  short-circuits to a fixed 911/988 message (`config.EMERGENCY_MESSAGE`) — the model
  never gets the chance to "handle" an emergency. Tuned for high recall.
- **No-advice floor.** A deterministic validator (`safety.contains_advice`) runs on the
  *final* answer regardless of which path produced it (LLM or deterministic) — a
  high-recall floor the deterministic path cannot skip. Advisory live-source prose is
  filtered at retrieval, and anything that still trips the validator is replaced with a
  deferral. (It's a regex heuristic tuned for recall, not an exhaustive classifier.)
- **Grounding & verbatim safety.** Every substantive sentence of an LLM answer must be
  supported by a tool result (a deterministic token-overlap check over medical passages
  + the patient's structured chart + tool summaries), and every allergy/alert must
  appear verbatim — otherwise the answer falls back to the grounded-by-construction
  deterministic one. Citations are attached in code from an allowlist of trusted domains
  (MedlinePlus / WHO) — **never authored by the LLM.**
- **No double-booking.** Slot booking is a single conditional `UPDATE ... WHERE
  status='available'` asserting `rowcount == 1` — race-safe, proven with two real
  threads (`test_no_double_booking_under_concurrency`).
- **No secrets committed.** `.env` is gitignored; the API key lives only locally.
  Parameterized SQL throughout; the data layer never raises into the request path.

## Known, intentional limitations (by design, for a demo)

Deliberate scope choices, documented here and in `WRITEUP.md` so they aren't mistaken
for oversights. A real deployment would close each one:

- **Auth is a demo directory.** Users and the password `demo123` are seeded in memory
  (`auth.py`); a real system uses an identity provider + authorization service.
  Passwords *are* hashed properly (PBKDF2-HMAC-SHA256, per-user salt, constant-time
  compare), but there is **no rate-limiting / lockout** and **no enforced session
  expiry**.
- **PHI at rest is not encrypted or redacted.** Chart data is stored in plaintext
  SQLite and shown in the UI. Redaction is best-effort and applied only at the **log**
  boundary (`observability.py` masks the message; record content is never logged).
  Production would encrypt at rest and run a real DLP pipeline at the storage boundary.
- **The emergency / no-advice gates are high-recall regex floors, not classifiers.**
  They are the deterministic floor; a production system would add an LLM classifier as
  defense-in-depth on top (never as a replacement).
- **SQLite, single-node.** Fine for a demo; production would use a managed database
  (the repository is the swap point) with access controls, audit retention, and backups.
- **No HIPAA/compliance posture.** No BAAs, access logging retention policy, or breach
  procedures — out of scope for a coursework demo.
