"""Central configuration for the Agentic Healthcare Assistant.

Everything tunable lives here: the model, resilience knobs, the role vocabulary, the
canonical safety strings (emergency message, disclaimer), the trusted-source
allowlist, and storage paths. Keys are read from the environment (loaded from a
local .env if present) — never hard-coded.

WHY a dedicated config module: in a healthcare system the safety-critical wording —
the emergency message a patient sees, the no-advice disclaimer, the trusted domains
we'll cite — must live in ONE auditable place, not scattered through logic. A
compliance reviewer can read every safety string here without reading any code.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # picks up .env in this folder if present; no-op otherwise

# --- Model -------------------------------------------------------------------
LLM_MODEL = "claude-sonnet-4-6"
LLM_MAX_TOKENS = 1024          # plans + summaries are larger than a single label
LLM_TIMEOUT = 30              # seconds before a call is abandoned
LLM_MAX_RETRIES = 2          # automatic backed-off retries on transient errors

# Whether the LLM is used for planning/phrasing. Off => deterministic stubs run, so
# the app + tests work with no API key. Set USE_LLM=0 to disable.
USE_LLM = os.getenv("USE_LLM", "1") not in ("0", "false", "False", "")

# --- Roles (the privilege model) --------------------------------------------
# Defined here so config + auth + handlers share one vocabulary. A patient sees only
# their own chart; an attendant/clinician acts on patients they're authorized for.
ROLE_PATIENT = "patient"
ROLE_ATTENDANT = "attendant"   # front-desk: books, adds records for assigned patients
ROLE_CLINICIAN = "clinician"   # adds clinical records/flags for assigned patients
ROLES = (ROLE_PATIENT, ROLE_ATTENDANT, ROLE_CLINICIAN)

# --- Database ----------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "healthcare.db")
DATABASE_URL = os.getenv("DATABASE_URL", "")  # postgres:// selects a Postgres backend

# --- Safety strings (the most important text in the system) ------------------
# Shown verbatim when the deterministic emergency gate fires. NEVER LLM-generated.
EMERGENCY_MESSAGE = (
    "⚠️ This may be a medical emergency. **If you are in danger, call 911 now** "
    "(or your local emergency number). For suicidal thoughts or a mental-health "
    "crisis, call or text **988** (US Suicide & Crisis Lifeline). This assistant "
    "cannot handle emergencies and is not a substitute for emergency services."
)

# Appended to every medical-information answer. The disclaimer is necessary but NOT
# sufficient — a deterministic no-advice validator (safety.py) is the real control.
MEDICAL_DISCLAIMER = (
    "_This is general health information, not medical advice. It may be incomplete "
    "or out of date. Always consult a qualified clinician about your situation._"
)

# Deferral message when the assistant declines (no grounding, advice-seeking, or a
# failed load-bearing step) — defer with a concrete next step.
DEFERRAL_MESSAGE = (
    "I can't safely answer that here. Please discuss it with your care team — "
    "I can help you book an appointment if that would be useful."
)

# --- Trusted sources (citations are pinned to these) -------------------------
# search_medical_info only fetches from / cites these domains. Citations are attached
# in code from the retrieval layer — never authored by the LLM.
ALLOWED_SOURCE_DOMAINS = ("medlineplus.gov", "www.who.int", "who.int")
SOURCE_STALENESS_DAYS = 365   # cached corpus older than this is flagged "as of [date]"
# Whether to attempt a LIVE MedlinePlus fetch before the offline corpus. Off in tests
# for determinism; the corpus is always the fallback. Set USE_LIVE_SEARCH=0 to disable.
USE_LIVE_SEARCH = os.getenv("USE_LIVE_SEARCH", "1") not in ("0", "false", "False", "")

# --- Appointments ------------------------------------------------------------
APPOINTMENT_STATUSES = ("available", "booked", "completed", "cancelled")
