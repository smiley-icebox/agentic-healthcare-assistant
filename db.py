"""Thin facade over the storage repository — a stable function-style API for the
tools, the graph, and the UI, so call sites don't depend on the repository class
directly (strangler-fig: a Postgres backend swaps in underneath without touching
callers). Authorization is enforced by the TOOLS via auth.can_access before these
are called; this layer is pure data access.
"""

from repository import get_repository


def init_db():
    get_repository()  # construction runs migrations


def reset_db():
    get_repository().reset()


# directory / seeding
def add_patient(patient_id, name, dob=None): get_repository().add_patient(patient_id, name, dob)
def add_doctor(doctor_id, name, specialty): get_repository().add_doctor(doctor_id, name, specialty)
def add_slot(doctor_id, start_at, slot_id=None): return get_repository().add_slot(doctor_id, start_at, slot_id)

# reads
def get_patient(patient_id): return get_repository().get_patient(patient_id)
def list_doctors(specialty=None): return get_repository().list_doctors(specialty)
def find_available_slots(specialty, limit=5): return get_repository().find_available_slots(specialty, limit)
def list_appointments(patient_id=None, doctor_id=None):
    return get_repository().list_appointments(patient_id, doctor_id)
def list_records(patient_id): return get_repository().list_records(patient_id)
def list_memory(patient_id, limit=20): return get_repository().list_memory(patient_id, limit)
def get_audit(entity_type, entity_id): return get_repository().get_audit(entity_type, entity_id)

# writes (transactional + audited in the repository)
def book_slot(slot_id, patient_id, specialty, actor):
    return get_repository().book_slot(slot_id, patient_id, specialty, actor)
def add_record(patient_id, record_type, label, value=None, note=None, recorded_by="system", supersedes=None):
    return get_repository().add_record(patient_id, record_type, label, value, note, recorded_by, supersedes)
def add_memory(patient_id, content): return get_repository().add_memory(patient_id, content)
