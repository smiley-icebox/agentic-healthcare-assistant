"""Authentication + authorization — where patient identity and access come from.

Two hard rules, both safety-critical because the payload here is a medical chart:

1. Identity is derived from a verified session, NEVER from client input or LLM/planner
   output. A message saying "pull records for patient P1002" can't move whose chart is
   read — the subject is fixed by the logged-in session.

2. `can_access(session, patient_id)` is DEFAULT-DENY. A patient may touch only their own
   chart; an attendant/clinician only the patients explicitly assigned to them (a
   *checked* list, never an implicit "staff sees everyone" bypass). Every record read
   and write must pass through it.

Passwords are PBKDF2-HMAC-SHA256 (per-user salt, constant-time compare). The user
directory is seeded in memory for the demo; a real system uses an identity provider
and an authorization service — see SECURITY.md.
"""

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from config import ROLE_ATTENDANT, ROLE_CLINICIAN, ROLE_PATIENT

_PBKDF2_ROUNDS = 200_000


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS).hex()


@dataclass(frozen=True)
class _User:
    user_id: str
    name: str
    role: str
    patient_id: str | None          # the user's own chart (patients only)
    assigned_patients: tuple         # charts a staff member may access
    salt: bytes
    pw_hash: str
    doctor_id: str | None = None     # the directory doctor a clinician maps to


@dataclass(frozen=True)
class Session:
    """A verified login. user_id/role/patient_id/assigned_patients are TRUSTED —
    derived from authentication. This is the ONLY source of "who am I / what may I
    touch"; tools and the planner never receive an identity from message content."""

    user_id: str
    name: str
    role: str
    username: str
    patient_id: str | None = None       # patient's own chart id (None for staff)
    assigned_patients: tuple = ()        # charts a staff member may access
    doctor_id: str | None = None         # directory doctor a clinician maps to (their calendar)
    issued_at: str = ""

    @property
    def is_staff(self) -> bool:
        return self.role in (ROLE_ATTENDANT, ROLE_CLINICIAN)


def can_access(session: Session, patient_id: str) -> bool:
    """Default-deny authorization for a patient chart. The single predicate every
    record read/write goes through."""
    if not session or not patient_id:
        return False
    if session.role == ROLE_PATIENT:
        return session.patient_id == patient_id
    if session.is_staff:
        return patient_id in session.assigned_patients
    return False


def _seed(username, password, user_id, name, role, patient_id=None, assigned=(), doctor_id=None):
    salt = os.urandom(16)
    return username, _User(user_id, name, role, patient_id, tuple(assigned), salt,
                           _hash_password(password, salt), doctor_id)


# Demo directory (passwords documented in README). Patients see only their own chart;
# staff are assigned an EXPLICIT patient list (a checked allowlist, not a bypass).
_USERS: dict[str, _User] = dict(
    [
        _seed("raj", "demo123", "U_raj", "Raj Patel", ROLE_PATIENT, patient_id="P1001"),
        _seed("maria", "demo123", "U_maria", "Maria Gomez", ROLE_PATIENT, patient_id="P1002"),
        _seed("alex", "demo123", "U_alex", "Alex (Front Desk)", ROLE_ATTENDANT,
              assigned=("P1001", "P1002")),
        _seed("drlee", "demo123", "U_drlee", "Dr. Lee", ROLE_CLINICIAN,
              assigned=("P1001", "P1002"), doctor_id="D_gp"),
    ]
)


def authenticate(username: str, password: str) -> Session | None:
    """Verify credentials and return a Session, or None on failure. Constant-time
    compare; a dummy hash runs for unknown users to avoid enumeration timing."""
    user = _USERS.get((username or "").strip().lower())
    if user is None:
        _hash_password(password or "", b"0" * 16)
        return None
    if not hmac.compare_digest(_hash_password(password or "", user.salt), user.pw_hash):
        return None
    return Session(
        user_id=user.user_id, name=user.name, role=user.role,
        username=username.strip().lower(), patient_id=user.patient_id,
        assigned_patients=user.assigned_patients, doctor_id=user.doctor_id,
        issued_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def demo_usernames() -> list[str]:
    return list(_USERS.keys())
