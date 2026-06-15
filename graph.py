"""The agent — plan-and-execute, SANDWICHED between deterministic safety gates.

Shape (the lesson carried from the prior project: code owns control flow + facts;
the LLM plans and phrases):

    message
      ▼ guard      — emergency red-flag gate (raw msg) + identity/scope (can_access)
      ▼ plan       — planner → validated-vs-allowlist steps (patient_id from session)
      ▼ execute    — run each tool; per-step status (ok/empty/error/denied); no fabrication
      ▼ synthesize — phrase a GROUNDED answer from step results; no-advice validator;
                     disclaimer + citations + escalation
    END

Emergency short-circuits before any LLM runs. A failed load-bearing step is never
papered over with model memory — the synthesizer surfaces it. patient_id is resolved
from the trusted session in `guard`, never taken from the plan or message.
"""

import time
from typing import Callable, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import config
import emergency
import knowledge
import llm
import memory
import observability
import planner
import safety
import tools
from auth import can_access

# The tool registry — the executor's allowlist of callables.
TOOL_REGISTRY: dict[str, Callable] = {
    "retrieve_history": tools.retrieve_history,
    "book_appointment": tools.book_appointment,
    "search_medical_info": tools.search_medical_info,
    "manage_records": tools.manage_records,
}


class State(TypedDict, total=False):
    message: str
    session: object
    subject_id: Optional[str]
    route: str                  # emergency | agent | deferral
    plan: list
    step_results: list
    answer: str
    citations: list
    escalated: bool
    grounded: bool
    fell_back: bool          # the LLM phrasing was rejected -> deterministic answer shipped
    advice_blocked: bool     # the no-advice floor fired -> answer replaced with a deferral


def _guard(state: State) -> dict:
    msg, session = state["message"], state["session"]
    # 1) Emergency FIRST, on the raw message — fixed message, no LLM, stop.
    if emergency.is_emergency(msg):
        return {"route": "emergency", "answer": config.EMERGENCY_MESSAGE, "escalated": True}
    # 2) Resolve the SUBJECT from the trusted session (never from the message/plan).
    subject = state.get("subject_id") or getattr(session, "patient_id", None)
    if not subject or not can_access(session, subject):
        return {"route": "deferral",
                "answer": "I can only act on a patient record you're authorized to access."}
    return {"route": "agent", "subject_id": subject}


def _route_after_guard(state: State) -> str:
    return "agent" if state.get("route") == "agent" else "stop"


def _make_plan_node(planner_fn: Callable):
    def plan_node(state: State) -> dict:
        plan = planner_fn(state["message"])
        if not plan:
            return {"route": "deferral", "plan": [], "answer": config.DEFERRAL_MESSAGE}
        return {"plan": plan}
    return plan_node


def _route_after_plan(state: State) -> str:
    return "execute" if state.get("plan") else "stop"


def _execute(state: State) -> dict:
    session, subject = state["session"], state["subject_id"]
    results = []
    for step in state["plan"]:
        fn = TOOL_REGISTRY.get(step["tool"])
        if fn is None:
            continue
        try:
            r = fn(session, subject, **step.get("args", {}))
        except Exception:
            r = {"status": "error", "summary": "", "data": {}, "citations": []}
        results.append({"tool": step["tool"], **r})
    return {"step_results": results}


def _deterministic_answer(query, results) -> tuple[str, list]:
    """Grounded-by-construction answer: tool summaries verbatim, failures surfaced,
    never fabricated. Used when USE_LLM is off (and as the safe baseline)."""
    parts, cites, has_info = [], [], False
    for r in results:
        st = r.get("status")
        if st == "ok":
            parts.append(r.get("summary", ""))
            cites += r.get("citations", [])
            if r["tool"] == "search_medical_info":
                has_info = True
        elif st == "empty":
            parts.append(f"({r.get('summary', 'nothing found for part of your request')})")
        elif st == "denied":
            parts.append(r.get("summary", "That action wasn't permitted."))
        else:  # error — surface, never paper over
            parts.append("I couldn't complete part of your request; please try again "
                         "or contact your care team.")
    answer = "  ".join(p for p in parts if p).strip() or config.DEFERRAL_MESSAGE
    if has_info:
        answer += "\n\n" + config.MEDICAL_DISCLAIMER
    return answer, cites


def _grounding_corpus(results) -> tuple[list, list]:
    """Build the corpus every LLM sentence must be supported by, plus the safety items
    that must appear verbatim. CRITICAL: this draws from retrieved medical passages AND
    the patient's STRUCTURED history (label/value) — so a history-only answer is grounded
    too, not just a medical-info answer. Allergies/alerts are collected as must-appear
    verbatim strings (a paraphrase or omission is direct harm)."""
    passages, verbatim = [], []
    for r in results:
        data = r.get("data", {})
        passages += data.get("passages", [])
        verbatim += data.get("verbatim_safety", [])
        for rec in data.get("records", []):                 # structured chart facts
            txt = rec.get("label", "") + (f" {rec['value']}" if rec.get("value") else "")
            if txt.strip():
                passages.append({"content": txt})
        # The tool's own summary is itself a grounded fact the LLM may restate — e.g. a
        # booking confirmation or a history line. Without it, legitimate transactional
        # sentences look "ungrounded." Grounding = "supported by a tool result," not
        # "supported by a medical passage."
        if r.get("status") == "ok" and r.get("summary"):
            passages.append({"content": r["summary"]})
    return passages, verbatim


def _llm_answer(query, results, mem) -> tuple[str, list, bool]:
    """LLM phrases an answer over the tool FACTS only, then three gates decide whether to
    ship it or fall back to the deterministic (grounded-by-construction) answer:
      1. no-advice   — advisory language is rejected;
      2. grounding   — every substantive sentence must be supported by the corpus
                       (medical passages + structured history);
      3. verbatim    — every safety item (allergy/alert) must appear verbatim.
    Returns (answer, citations, fell_back) where fell_back=True means the LLM phrasing
    was rejected and the deterministic answer is being shipped instead."""
    passages, verbatim = _grounding_corpus(results)
    facts = "\n".join(f"- [{r['tool']}/{r['status']}] {r.get('summary','')}" for r in results)
    ctx = ("\nKnown patient context: " + "; ".join(mem)) if mem else ""
    system = (
        "You are a healthcare assistant. Using ONLY the tool results below, write a short, "
        "warm reply in PLAIN PROSE — no headings, bullet lists, markdown, or your own "
        "disclaimer (a disclaimer is appended automatically), and no preamble like 'here "
        "is what I found'. Cover the facts given but add NO medical facts not present. "
        "State any allergy or safety alert exactly as written. Give information, NEVER "
        "advice or a diagnosis — no 'you should', no dosing, no telling the patient what "
        "to do. If a step failed, say so plainly. Do not invent citations.")
    try:
        msg = llm.chat_model(700, temperature=0.2).invoke(
            [("system", system), ("human", f"Request: {query}\n\nTool results:\n{facts}{ctx}")])
        text = llm.extract_text(getattr(msg, "content", "")).strip()
    except Exception:
        text = ""
    det, cites = _deterministic_answer(query, results)
    if not text:
        return det, cites, True
    if safety.contains_advice(text):
        return det, cites, True
    # Grounding gate: every substantive sentence supported by the corpus (now incl. history).
    if passages:
        for sent in [s for s in text.replace("\n", " ").split(". ") if len(s.split()) > 4]:
            if not knowledge.is_grounded(sent, passages):
                return det, cites, True
    # Verbatim-safety gate: a softened/dropped allergy or alert is direct harm. The
    # allergen/alert label must be present (case-insensitive — "penicillin" is fine, a
    # dropped or paraphrased-away "Penicillin" is not).
    low = text.lower()
    for item in verbatim:
        if item and item.lower() not in low:
            return det, cites, True
    if any(r["tool"] == "search_medical_info" and r["status"] == "ok" for r in results):
        text += "\n\n" + config.MEDICAL_DISCLAIMER
    return text, cites, False


def _synthesize(state: State) -> dict:
    results = state.get("step_results", [])
    query = state["message"]
    if config.USE_LLM:
        answer, cites, fell_back = _llm_answer(query, results, memory.load_memory(state["subject_id"]))
    else:
        answer, cites, fell_back = *_deterministic_answer(query, results), False
    # Universal no-advice floor: the validator runs on the FINAL answer regardless of
    # which path produced it. The LLM path already self-checks, but the deterministic
    # path quotes tool/source text verbatim — so this is a high-recall backstop that the
    # deterministic path cannot bypass. If it fires, we defer rather than ship.
    advice_blocked = False
    if safety.contains_advice(answer):
        answer, cites, advice_blocked = config.DEFERRAL_MESSAGE, [], True
    grounded = not fell_back and not advice_blocked
    # de-dup citations by url
    seen, uniq = set(), []
    for c in cites:
        if c.get("url") and c["url"] not in seen:
            seen.add(c["url"]); uniq.append(c)
    # persist a short, redacted memory note
    memory.save_memory(state["subject_id"],
                       f"Request handled via {[r['tool'] for r in results]}")
    return {"answer": answer, "citations": uniq, "grounded": grounded,
            "fell_back": fell_back, "advice_blocked": advice_blocked}


def build_graph(planner_fn: Optional[Callable] = None):
    pf = planner_fn or planner.make_plan
    g = StateGraph(State)
    g.add_node("guard", _guard)
    g.add_node("plan", _make_plan_node(pf))
    g.add_node("execute", _execute)
    g.add_node("synthesize", _synthesize)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", _route_after_guard, {"agent": "plan", "stop": END})
    g.add_conditional_edges("plan", _route_after_plan, {"execute": "execute", "stop": END})
    g.add_edge("execute", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_graph = None


def _default_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def respond(message: str, session, subject_id: Optional[str] = None, graph=None) -> dict:
    """Single entry point: run the agent, write a PHI-redacted trace, return the result."""
    g = graph if graph is not None else _default_graph()
    started = time.perf_counter()
    final = g.invoke({"message": message, "session": session, "subject_id": subject_id})
    latency = round((time.perf_counter() - started) * 1000, 1)
    results = final.get("step_results", []) or []
    observability.record(
        message=message, route=final.get("route"),
        plan=[s["tool"] for s in final.get("plan", [])],
        tools=[{"tool": r["tool"], "status": r["status"]} for r in results],
        success=bool(final.get("answer")), escalated=final.get("escalated"),
        grounded=final.get("grounded"), latency_ms=latency,
        fell_back=final.get("fell_back"), advice_blocked=final.get("advice_blocked"),
    )
    final["latency_ms"] = latency
    return final
