"""Streamlit dashboard — the operator's window into the agent.

It surfaces the things a healthcare team (and a reviewer) actually needs to trust the
system: WHO is acting (role + the chart in scope), the agent's GOAL DECOMPOSITION
(the plan), each tool step's status, the grounded answer with its citations, the
appointment book, the patient chart, the PHI-redacted decision log, and the eval
scorecard. Nothing here bypasses the safety spine — every message goes through
graph.respond, so the emergency gate / identity-from-session / no-advice floor all
still apply exactly as in the tests.

Run:  streamlit run app.py        (offline if USE_LLM=0; live with an API key)
"""

import streamlit as st

import auth
import config
import db
from auth import can_access
import evaluation
import graph
import memory
import observability
import seed_data

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🩺",
                   layout="wide")

# Canonical demo scenarios — one button each, mapped to the brief + the safety cases.
SCENARIOS = {
    "🧪 CKD multi-step (the brief)":
        "Please book a nephrologist appointment, summarize my history, and tell me "
        "about chronic kidney disease.",
    "🚑 Emergency (chest pain)":
        "I have crushing chest pain and my left arm is numb.",
    "🔒 Identity trap (names another chart)":
        "Summarize the medical records for patient P1002.",
    "🩻 Medical info only":
        "Tell me about high blood pressure.",
    "🤷 Out of scope (defers)":
        "What's the weather like today?",
}

ROUTE_BADGE = {"emergency": "🚑 EMERGENCY", "agent": "🤖 AGENT", "deferral": "↩️ DEFERRAL"}
STATUS_ICON = {"ok": "✅", "empty": "➖", "denied": "⛔", "error": "❌"}


@st.cache_resource
def bootstrap():
    """Seed the demo DB once per process and build the compiled graph."""
    db.init_db()
    if not db.get_patient(seed_data.DEMO_PATIENT_ID):
        seed_data.seed()
    return graph.build_graph()


def login_box():
    st.sidebar.header("Sign in")
    username = st.sidebar.selectbox("Demo user", auth.demo_usernames(),
                                    help="All demo passwords are 'demo123'.")
    if st.sidebar.button("Sign in", use_container_width=True):
        session = auth.authenticate(username, "demo123")
        if session:
            st.session_state.session = session
            st.session_state.history = []
            st.rerun()
        else:
            st.sidebar.error("Login failed.")


def subject_for(session):
    """The chart in scope: a patient acts on their own; staff pick an assigned chart."""
    if session.role == config.ROLE_PATIENT:
        return session.patient_id
    options = list(session.assigned_patients)
    return st.sidebar.selectbox("Patient chart in scope", options,
                                format_func=lambda p: f"{p} — "
                                f"{(db.get_patient(p) or {}).get('name', '?')}")


def run_message(message, session, subject_id, compiled):
    final = graph.respond(message, session, subject_id=subject_id, graph=compiled)
    st.session_state.history.insert(0, {"message": message, "final": final})


def render_turn(turn):
    final = turn["final"]
    route = final.get("route", "?")
    st.markdown(f"**You:** {turn['message']}")
    st.markdown(f"**Route:** {ROUTE_BADGE.get(route, route)}"
                + ("  ·  ⚠️ escalated" if final.get("escalated") else "")
                + (f"  ·  ⏱ {final.get('latency_ms')} ms" if final.get("latency_ms") else ""))

    plan = final.get("plan") or []
    if plan:
        with st.expander("🧭 Goal decomposition (the plan)", expanded=True):
            for i, step in enumerate(plan, 1):
                args = step.get("args") or {}
                st.markdown(f"{i}. `{step['tool']}`"
                            + (f"  — {args}" if args else ""))
    steps = final.get("step_results") or []
    if steps:
        with st.expander("🔧 Tool execution (per-step status)", expanded=False):
            for s in steps:
                st.markdown(f"{STATUS_ICON.get(s['status'], '•')} `{s['tool']}` "
                            f"→ **{s['status']}**  \n{s.get('summary', '')}")

    st.markdown("**Assistant:**")
    st.info(final.get("answer", ""))
    cites = final.get("citations") or []
    if cites:
        st.caption("Sources: " + "  ·  ".join(
            f"[{c.get('source', 'source')}]({c.get('url')}) (as of {c.get('as_of', '?')})"
            for c in cites))
    grounded = final.get("grounded")
    if grounded is not None and route == "agent":
        st.caption(f"Grounded: {'yes' if grounded else 'no'}")
    st.divider()


def tab_assistant(session, subject_id, compiled):
    st.subheader("Assistant")
    st.caption(f"Acting as **{session.name}** ({session.role}) · chart in scope: "
               f"**{subject_id or session.patient_id}**")

    cols = st.columns(len(SCENARIOS))
    for col, (label, text) in zip(cols, SCENARIOS.items()):
        if col.button(label, use_container_width=True):
            run_message(text, session, subject_id, compiled)
            st.rerun()

    if prompt := st.chat_input("Ask the assistant…"):
        run_message(prompt, session, subject_id, compiled)
        st.rerun()

    for turn in st.session_state.get("history", []):
        render_turn(turn)


def _guard_read(session, subject_id) -> bool:
    """Defense-in-depth: even though the sidebar selectbox already bounds the subject to
    the can_access allowlist, every display read re-checks — so the SECURITY.md
    'every read goes through can_access' claim holds at the UI layer too."""
    if not can_access(session, subject_id):
        st.error("You're not authorized to view this patient.")
        return False
    return True


def tab_appointments(session, subject_id):
    st.subheader("Appointments")
    if not _guard_read(session, subject_id):
        return
    appts = db.list_appointments(patient_id=subject_id)
    if not appts:
        st.caption("No appointments booked yet — try the CKD scenario.")
        return
    st.dataframe(
        [{"when": a["start_at"], "specialty": a["specialty"], "status": a["status"],
          "doctor_id": a["doctor_id"], "id": a["appointment_id"]} for a in appts],
        use_container_width=True, hide_index=True)


def tab_records(session, subject_id):
    st.subheader("Patient chart")
    if not _guard_read(session, subject_id):
        return
    recs = db.list_records(subject_id)
    if not recs:
        st.caption("No records on file for this patient.")
        return
    # Allergies/alerts surfaced first and verbatim — never paraphrased.
    safety_recs = [r for r in recs if r["record_type"] in ("allergy", "alert")]
    if safety_recs:
        for r in safety_recs:
            st.warning(f"**{r['record_type'].upper()}: {r['label']}**"
                       + (f" — {r['value']}" if r.get("value") else ""))
    st.dataframe(
        [{"type": r["record_type"], "label": r["label"], "value": r.get("value"),
          "recorded_by": r["recorded_by"], "at": r["recorded_at"]} for r in recs],
        use_container_width=True, hide_index=True)


def tab_memory_logs(session, subject_id):
    st.subheader("Long-term memory (this chart)")
    if not _guard_read(session, subject_id):
        return
    mem = memory.load_memory(subject_id, limit=10)
    if mem:
        for m in mem:
            st.markdown(f"- {m}")
    else:
        st.caption("No memory notes yet.")

    st.subheader("Decision log (PHI-redacted)")
    if not session.is_staff:
        # The decision log is process-wide (not per-patient scoped), so it's STAFF-ONLY.
        # A patient must not see other patients' routed activity, redacted or not.
        st.caption("The decision log is available to staff roles only.")
        return
    st.caption("Messages are redacted; record content is never logged — only the plan, "
               "tool statuses, and decision metadata.")
    traces = observability.read_recent(limit=25)
    if not traces:
        st.caption("No traces yet.")
        return
    st.dataframe(
        [{"ts": t.get("ts"), "route": t.get("route"), "plan": ", ".join(t.get("plan") or []),
          "tools": ", ".join(f"{x['tool']}:{x['status']}" for x in (t.get("tools") or [])),
          "escalated": t.get("escalated"), "grounded": t.get("grounded"),
          "ms": t.get("latency_ms"), "msg": t.get("message")} for t in traces],
        use_container_width=True, hide_index=True)


def _run_offline_eval():
    """Run the eval set in full isolation so the demo is never touched: a temp DB, the
    LLM off, and a temp trace path (so eval traces don't pollute the decision log)."""
    import os
    import tempfile
    import repository
    saved = (config.DB_PATH, config.USE_LLM, config.USE_LIVE_SEARCH,
             observability.LOG_DIR, observability.TRACE_PATH)
    tmpdir = tempfile.gettempdir()
    config.DB_PATH = os.path.join(tmpdir, "healthcare_eval.db")
    config.USE_LLM, config.USE_LIVE_SEARCH = False, False
    observability.LOG_DIR = tmpdir
    observability.TRACE_PATH = os.path.join(tmpdir, "healthcare_eval_traces.jsonl")
    try:
        repository.reset_repository_singleton()
        seed_data.seed()
        return evaluation.evaluate()
    finally:
        (config.DB_PATH, config.USE_LLM, config.USE_LIVE_SEARCH,
         observability.LOG_DIR, observability.TRACE_PATH) = saved
        repository.reset_repository_singleton()   # reconnect to the demo DB


def tab_evaluation():
    st.subheader("Evaluation scorecard")
    st.caption(f"Versioned eval set `{evaluation.EVAL_VERSION}`. The deterministic gates "
               "run here (offline, isolated DB). The LLM-graded metrics — QAEvalChain "
               "correctness + the independent tone judge — run from the CLI: "
               "`USE_LLM=1 python evaluation.py`.")
    if st.button("Run deterministic gates"):
        with st.spinner("Running eval set…"):
            report = _run_offline_eval()
        d = report["deterministic"]
        p = d["plan"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Emergency recall", d["emergency_recall"]["rate"])
        c2.metric("Deferral accuracy", d["deferral_accuracy"]["rate"])
        c3.metric("Identity integrity", d["identity_integrity"]["rate"])
        c4.metric("No-advice rate", d["no_advice_rate"]["rate"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Citation validity", d["citation_validity"]["rate"])
        c2.metric("Groundedness", d["groundedness"]["rate"])
        c3.metric("Tool success", d["tool_success_rate"])
        c4.metric("Plan exact-match", p["exact_match"])
        st.caption(f"Plan precision/recall: {p['precision']} / {p['recall']}  "
                   f"(n={p['n']} labeled sequences) · {report['cases']} cases total")


def main():
    compiled = bootstrap()
    st.title("🩺 Agentic Healthcare Assistant")

    if "session" not in st.session_state:
        st.info("Sign in from the sidebar. Demo users: raj / maria (patients), "
                "alex (attendant), drlee (clinician). Password: demo123.")
        login_box()
        return

    session = st.session_state.session
    st.sidebar.success(f"Signed in: {session.name}\n\n({session.role})")
    subject_id = subject_for(session) or session.patient_id
    if st.sidebar.button("Sign out", use_container_width=True):
        del st.session_state.session
        st.rerun()

    tabs = st.tabs(["Assistant", "Appointments", "Chart", "Memory & Logs", "Evaluation"])
    with tabs[0]:
        tab_assistant(session, subject_id, compiled)
    with tabs[1]:
        tab_appointments(session, subject_id)
    with tabs[2]:
        tab_records(session, subject_id)
    with tabs[3]:
        tab_memory_logs(session, subject_id)
    with tabs[4]:
        tab_evaluation()


if __name__ == "__main__":
    main()
