"""Streamlit app — a CHAT-FIRST experience with the agent's machinery one click away.

The earlier version was a reviewer's cockpit (five tabs of plan/tools/logs/eval). This
one leads with the conversation a real user would have, and tucks the rubric-proving
internals into an inline "How I worked this out" panel per reply plus an "Under the
hood" sidebar section. Nothing about the safety spine changes — every message still
goes through graph.respond, so the emergency gate / identity-from-session / grounding /
no-advice floor all apply exactly as in the tests.

Run:  streamlit run app.py        (offline if USE_LLM=0; live with an API key)
"""

import streamlit as st

import auth
import config
import db
import evaluation
import graph
import memory
import observability
import seed_data
from auth import can_access

st.set_page_config(page_title="Health Assistant", page_icon="🩺", layout="centered")

# Demo roster — one-click sign-in (all passwords are demo123, documented in README).
DEMO_USERS = [
    ("raj",   "🧑 Raj Patel",    "Patient · 50, chronic kidney disease"),
    ("maria", "🧑 Maria Gomez",  "Patient · no history on file"),
    ("alex",  "🧑‍💼 Alex",        "Front-desk attendant"),
    ("drlee", "🩺 Dr. Lee",      "Clinician"),
]

# Role-aware quick starts. Each chip is (label, message-sent-to-the-agent).
PATIENT_CHIPS = [
    ("✨ Book + history + learn (CKD)",
     "Please book a nephrologist appointment, summarize my history, and tell me about "
     "chronic kidney disease."),
    ("📋 Summarize my history", "Summarize my medical history."),
    ("💊 Tell me about CKD", "Tell me about chronic kidney disease."),
    ("🚑 Emergency (demo)", "I have crushing chest pain and my left arm is numb."),
]
STAFF_CHIPS = [
    ("📋 Summarize this patient's history", "Summarize this patient's history."),
    ("📅 Book a cardiology appointment", "Book a cardiology appointment."),
    ("💊 Look up high blood pressure", "Tell me about high blood pressure."),
]

ROUTE_BADGE = {"emergency": "🚑 Emergency", "agent": "🤖 Agent", "deferral": "↩️ Deferral"}
STATUS_ICON = {"ok": "✅", "empty": "➖", "denied": "⛔", "error": "❌"}


@st.cache_resource
def bootstrap():
    """Seed the demo DB once per process and build the compiled graph."""
    db.init_db()
    if not db.get_patient(seed_data.DEMO_PATIENT_ID):
        seed_data.seed()
    return graph.build_graph()


# --- auth helpers ------------------------------------------------------------
def sign_in(username):
    session = auth.authenticate(username, "demo123")
    if session:
        st.session_state.session = session
        st.session_state.history = []
        st.rerun()


def login_screen():
    st.title("🩺 Health Assistant")
    st.caption("A demo agentic assistant on synthetic data — it books appointments, "
               "explains your records, and looks up trusted health info. Choose who to "
               "sign in as:")
    cols = st.columns(2)
    for i, (uname, label, sub) in enumerate(DEMO_USERS):
        with cols[i % 2]:
            if st.button(f"{label}\n\n{sub}", use_container_width=True, key=f"login_{uname}"):
                sign_in(uname)
    st.caption("All demo passwords are `demo123`. Not a medical device — gives "
               "information, never advice, and escalates emergencies to 911/988.")


# --- the conversation --------------------------------------------------------
def run_message(message, session, subject_id, compiled):
    final = graph.respond(message, session, subject_id=subject_id, graph=compiled)
    st.session_state.history.append({"message": message, "final": final})


def render_inspector(final):
    """The inline, collapsed 'how I worked this out' — the rubric machinery, contextual."""
    route = final.get("route", "?")
    with st.expander("🔍 How I worked this out", expanded=False):
        bits = [f"**Route:** {ROUTE_BADGE.get(route, route)}"]
        if final.get("latency_ms") is not None:
            bits.append(f"⏱ {final['latency_ms']} ms")
        if final.get("escalated"):
            bits.append("⚠️ escalated")
        st.markdown("  ·  ".join(bits))

        plan = final.get("plan") or []
        if plan:
            st.markdown("**Goal decomposition (the plan):**")
            for i, step in enumerate(plan, 1):
                args = step.get("args") or {}
                st.markdown(f"{i}. `{step['tool']}`" + (f" — {args}" if args else ""))
        steps = final.get("step_results") or []
        if steps:
            st.markdown("**Tool execution:**")
            for s in steps:
                st.markdown(f"{STATUS_ICON.get(s['status'], '•')} `{s['tool']}` → "
                            f"**{s['status']}** — {s.get('summary', '')}")
        if route == "agent":
            flags = [f"grounded: {'yes' if final.get('grounded') else 'no'}"]
            if final.get("fell_back"):
                flags.append("LLM phrasing rejected → safe deterministic answer")
            if final.get("advice_blocked"):
                flags.append("no-advice floor fired → deferral")
            st.caption(" · ".join(flags))


def render_turn(turn):
    with st.chat_message("user"):
        st.markdown(turn["message"])
    final = turn["final"]
    with st.chat_message("assistant"):
        st.markdown(final.get("answer", ""))
        cites = final.get("citations") or []
        if cites:
            st.caption("Sources: " + "  ·  ".join(
                f"[{c.get('source', 'source')}]({c.get('url')}) (as of {c.get('as_of', '?')})"
                for c in cites))
        render_inspector(final)


def chat_area(session, subject_id, compiled):
    if session.is_staff:
        intro = (f"Hi {session.name}. You're acting on **{subject_id}** "
                 f"({(db.get_patient(subject_id) or {}).get('name', '?')}). I can pull "
                 "their history, book appointments, add records, or look up conditions.")
        chips = STAFF_CHIPS
    else:
        intro = (f"Hi {session.name.split()[0]}. I can book appointments, explain what's "
                 "on your record, or look up trusted health info. What would you like to do?")
        chips = PATIENT_CHIPS

    with st.chat_message("assistant"):
        st.markdown(intro)

    # Persistent quick-starts (discoverability — the old UI gave no hint what to try).
    st.caption("Try one:")
    cols = st.columns(2)
    pending = None
    for i, (label, text) in enumerate(chips):
        if cols[i % 2].button(label, use_container_width=True, key=f"chip_{i}"):
            pending = text

    for turn in st.session_state.get("history", []):
        render_turn(turn)

    typed = st.chat_input("Ask the assistant…")
    msg = typed or pending
    if msg:
        run_message(msg, session, subject_id, compiled)
        st.rerun()


# --- sidebar: account + glance + inspector -----------------------------------
def sidebar(session):
    st.sidebar.markdown(f"**{session.name}**  \n{session.role}")
    with st.sidebar.expander("🔄 Switch user"):
        for uname, label, _ in DEMO_USERS:
            if st.button(label, use_container_width=True, key=f"switch_{uname}"):
                sign_in(uname)
    if st.sidebar.button("Sign out", use_container_width=True):
        del st.session_state.session
        st.rerun()

    # Staff pick the patient in focus; a patient is always their own chart.
    if session.is_staff:
        subject_id = st.sidebar.selectbox(
            "Patient in focus", list(session.assigned_patients),
            format_func=lambda p: f"{p} — {(db.get_patient(p) or {}).get('name', '?')}")
    else:
        subject_id = session.patient_id

    # Changing the patient in focus must NOT replay the prior patient's transcript under
    # the new patient's header — a misidentification hazard. Clear it on subject change.
    if st.session_state.get("_subject") != subject_id:
        st.session_state.history = []
        st.session_state._subject = subject_id

    st.sidebar.divider()
    _doctor_schedule(session)
    _sidebar_glance(session, subject_id)
    _sidebar_under_the_hood(session)
    return subject_id


def _doctor_schedule(session):
    """Doctor view: a clinician's own calendar (appointments booked WITH them)."""
    doctor_id = getattr(session, "doctor_id", None)
    if not doctor_id:
        return
    st.sidebar.markdown("### 🩺 My schedule")
    appts = db.list_appointments(doctor_id=doctor_id)
    if appts:
        for a in appts:
            st.sidebar.caption(f"{a['start_at'][:16].replace('T', ' ')} · "
                               f"{a['specialty']} · {a['patient_id']} · {a['status']}")
    else:
        st.sidebar.caption("No appointments on your calendar.")
    st.sidebar.divider()


def _sidebar_glance(session, subject_id):
    if not can_access(session, subject_id):
        return
    st.sidebar.markdown("### 📅 Appointments")
    appts = db.list_appointments(patient_id=subject_id)
    if appts:
        for a in appts:
            st.sidebar.caption(f"{a['start_at'][:16].replace('T', ' ')} · "
                               f"{a['specialty']} · {a['status']}")
    else:
        st.sidebar.caption("None yet.")

    st.sidebar.markdown("### 📋 Chart")
    recs = db.list_records(subject_id)
    safety_recs = [r for r in recs if r["record_type"] in ("allergy", "alert")]
    for r in safety_recs:        # safety flags surfaced first, verbatim
        st.sidebar.warning(f"**{r['record_type'].upper()}: {r['label']}**"
                           + (f" — {r['value']}" if r.get("value") else ""))
    others = [r for r in recs if r["record_type"] not in ("allergy", "alert")]
    if others:
        with st.sidebar.expander(f"Full chart ({len(recs)} entries)"):
            for r in recs:
                st.caption(f"{r['record_type']}: {r['label']}"
                           + (f" — {r['value']}" if r.get("value") else ""))
    elif not safety_recs:
        st.sidebar.caption("No records on file.")

    # Agent memory traces (the FAISS-backed long-term patient summaries), PHI-redacted.
    st.sidebar.markdown("### 🧠 Long-term memory")
    mem = memory.load_memory(subject_id, limit=6)
    if mem:
        for m in mem:
            st.sidebar.caption(f"• {m}")
    else:
        st.sidebar.caption("No memory yet — it builds as you interact.")


def _sidebar_under_the_hood(session):
    st.sidebar.divider()
    with st.sidebar.expander("🔍 Under the hood"):
        if st.button("Run evaluation", use_container_width=True):
            with st.spinner("Running eval set…"):
                st.session_state.eval_report = _run_offline_eval()
        rep = st.session_state.get("eval_report")
        if rep:
            d = rep["deterministic"]
            st.caption(f"Eval `{rep['version']}` · {rep['cases']} cases")
            for k in ("emergency_recall", "deferral_accuracy", "identity_integrity",
                      "no_advice_rate", "citation_validity", "groundedness"):
                st.caption(f"{k}: {d[k]['rate']}")
            st.caption(f"tool_success: {d['tool_success_rate']} · "
                       f"plan recall: {d['plan']['recall']}")
        st.divider()
        if session.is_staff:
            st.caption("**Decision log** (PHI-redacted, staff only)")
            for t in observability.read_recent(limit=8):
                st.caption(f"{(t.get('ts') or '')[11:19]} · {t.get('route')} · "
                           f"{', '.join(t.get('plan') or []) or '—'}")
        else:
            st.caption("The decision log is staff-only.")


def _run_offline_eval():
    """Run the deterministic gates offline. evaluation.evaluate() is self-isolating (temp
    DB + temp traces, restored after), so the demo DB/log are never touched; here we only
    pin USE_LLM off so the panel stays fast and key-free. NOTE: it briefly repoints
    process-global config, so don't run it concurrently with a live chat turn in another
    session — fine for this single-user demo."""
    saved_llm = config.USE_LLM
    config.USE_LLM = False
    try:
        return evaluation.evaluate()
    finally:
        config.USE_LLM = saved_llm


def main():
    compiled = bootstrap()
    if "session" not in st.session_state:
        login_screen()
        return
    session = st.session_state.session
    subject_id = sidebar(session)
    st.title("🩺 Health Assistant")
    chat_area(session, subject_id, compiled)


if __name__ == "__main__":
    main()
