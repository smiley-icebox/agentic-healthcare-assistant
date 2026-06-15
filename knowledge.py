"""RAG over trusted medical sources — offline-first, citation-pinned.

search() returns retrieved passages + CITATIONS attached IN CODE (source URL +
as-of date) — the LLM never authors a citation. Order: try a LIVE MedlinePlus fetch
(allowlisted domain) when enabled, else fall back to a small curated offline corpus
indexed in FAISS. The corpus is the source of truth; the FAISS index is rebuilt from
it at startup (no drift), with lightweight TF-IDF vectors (no torch) so retrieval is
deterministic and offline.

Grounding is enforced separately at synthesis (safety.py / the graph): every sentence
shown to a patient must be supported by a retrieved passage. Here we just retrieve and
cite.
"""

import re
from datetime import datetime, timezone

import config
import safety

# --- Curated offline corpus (the trusted fallback) ---------------------------
# Each entry is a real MedlinePlus health-topic summary, stamped with its source +
# the date it was captured (surfaced as "as of [date]"). Kept small on purpose.
CORPUS = [
    {
        "topic": "Chronic kidney disease",
        "tags": "chronic kidney disease ckd kidney renal nephrology dialysis kidneys",
        "content": ("Chronic kidney disease (CKD) means the kidneys are damaged and "
                    "can't filter blood well. Common causes are diabetes and high blood "
                    "pressure. Management focuses on controlling blood pressure and blood "
                    "sugar, a kidney-friendly diet, and avoiding medicines that harm the "
                    "kidneys such as NSAIDs. Advanced CKD may require dialysis or a transplant."),
        "url": "https://medlineplus.gov/chronickidneydisease.html",
        "as_of": "2026-01-15",
    },
    {
        "topic": "High blood pressure",
        "tags": "high blood pressure hypertension bp cardiovascular",
        "content": ("High blood pressure (hypertension) usually has no symptoms but raises "
                    "the risk of heart disease, stroke, and kidney disease. It is managed "
                    "with lifestyle changes (reduced salt, exercise, weight management) and, "
                    "when needed, blood-pressure medicines."),
        "url": "https://medlineplus.gov/highbloodpressure.html",
        "as_of": "2026-01-15",
    },
    {
        "topic": "Diabetes",
        "tags": "diabetes type 2 blood sugar glucose insulin a1c",
        "content": ("Diabetes is a condition of high blood glucose. Type 2 is the most "
                    "common. Management includes healthy eating, physical activity, blood-"
                    "glucose monitoring, and medicines such as metformin or insulin when "
                    "prescribed."),
        "url": "https://medlineplus.gov/diabetes.html",
        "as_of": "2026-01-15",
    },
    {
        "topic": "Penicillin allergy",
        "tags": "penicillin allergy antibiotic rash anaphylaxis drug allergy",
        "content": ("A penicillin allergy is an immune reaction to penicillin antibiotics, "
                    "ranging from rash to, rarely, anaphylaxis. People with a documented "
                    "penicillin allergy are usually given alternative antibiotics."),
        "url": "https://medlineplus.gov/penicillin.html",
        "as_of": "2026-01-15",
    },
]

_WORD_RE = re.compile(r"[a-z0-9]+")


class _Retriever:
    """TF-IDF vectors + a FAISS inner-product index over the corpus. Built lazily and
    rebuilt from CORPUS (the source of truth) — never persisted, so it can't drift."""

    def __init__(self):
        self._built = False

    def _build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import faiss
        docs = [f"{e['topic']} {e['tags']} {e['content']}" for e in CORPUS]
        self._vec = TfidfVectorizer(stop_words="english")
        mat = self._vec.fit_transform(docs).astype("float32").toarray()
        faiss.normalize_L2(mat)                      # cosine via inner product
        self._index = faiss.IndexFlatIP(mat.shape[1])
        self._index.add(mat)
        self._built = True

    def retrieve(self, query, k=2, min_score=0.05):
        if not self._built:
            self._build()
        import faiss
        q = self._vec.transform([query or ""]).astype("float32").toarray()
        if q.sum() == 0:
            return []
        faiss.normalize_L2(q)
        scores, idx = self._index.search(q, min(k, len(CORPUS)))
        hits = []
        for score, i in zip(scores[0], idx[0]):
            if i >= 0 and score >= min_score:
                hits.append(CORPUS[i])
        return hits


_retriever = _Retriever()


def _citation(entry) -> dict:
    return {"source": entry["topic"], "url": entry["url"], "as_of": entry["as_of"]}


def _stale(as_of: str) -> bool:
    try:
        d = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - d).days > config.SOURCE_STALENESS_DAYS
    except ValueError:
        return False


def _live_fetch(condition):
    """Best-effort MedlinePlus health-topics search (allowlisted). Returns a passage
    dict or None. Never raises; falls through to the corpus on any failure."""
    if not config.USE_LIVE_SEARCH:
        return None
    try:
        import requests
        from xml.etree import ElementTree as ET
        resp = requests.get(
            "https://wsearch.nlm.nih.gov/ws/query",
            params={"db": "healthTopics", "term": condition, "retmax": 1},
            timeout=6,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        doc = root.find(".//document")
        if doc is None:
            return None
        url = doc.get("url", "")
        if not any(dom in url for dom in config.ALLOWED_SOURCE_DOMAINS):
            return None  # provenance check: only cite allowlisted domains
        def field(name):
            el = doc.find(f".//content[@name='{name}']")
            return re.sub(r"<[^>]+>", "", el.text or "") if el is not None else ""
        summary = field("FullSummary") or field("snippet")
        title = field("title") or condition
        if not summary:
            return None
        today = datetime.now(timezone.utc).date().isoformat()
        return {"topic": title, "content": summary.strip(), "url": url,
                "as_of": today, "as_of_note": f" (as of {today})"}
    except Exception:
        return None


def search(condition: str) -> dict:
    """Return {passages, citations, live} for a condition. Live MedlinePlus first
    (when enabled), else the curated FAISS corpus; always cited.

    Live source prose is written in the second person ("your kidneys filter your
    blood") and so reads as advice when quoted verbatim — and the deterministic
    answer DOES quote verbatim. So a live passage is admitted only if it passes the
    no-advice validator; otherwise we fall back to the curated corpus, which is
    written advice-free on purpose. This applies the no-advice floor at the live-RAG
    boundary too, while preserving live retrieval for non-advisory sources."""
    live = _live_fetch(condition)
    if live and not safety.contains_advice(live["content"]):
        return {"passages": [live], "citations": [_citation(live)], "live": True}
    hits = _retriever.retrieve(condition, k=2)
    passages = []
    for h in hits:
        note = f" (as of {h['as_of']})" + ("  [may be out of date]" if _stale(h["as_of"]) else "")
        passages.append({**h, "content": h["content"], "as_of_note": note})
    return {"passages": passages, "citations": [_citation(h) for h in hits], "live": False}


def _tokens(text):
    return {t for t in _WORD_RE.findall((text or "").lower()) if len(t) >= 4}


# Calibrated from real output: a FAITHFUL paraphrase of the corpus scores ~0.27-0.6
# substantive-token overlap (connective words like "despite this" dilute it), while an
# off-topic FABRICATION scores ~0.0. 0.25 sits between the two — it admits honest
# paraphrase and rejects fabrication. Tuned low on purpose: a false reject only costs
# fluency (we fall back to the grounded deterministic answer), so the safe error is to
# under-ship LLM text, not to ship an ungrounded claim. (Token overlap can't catch a
# contradiction that reuses on-topic words — that needs entailment; see SECURITY.md.)
GROUNDING_MIN_OVERLAP = 0.25


def is_grounded(sentence: str, passages: list, min_overlap: float = GROUNDING_MIN_OVERLAP) -> bool:
    """Deterministic claim-grounding check: a sentence is grounded if enough of its
    substantive tokens appear in the retrieved passages. A lightweight, offline, non-LLM
    filter; the graph applies it to every sentence before shipping LLM-phrased text."""
    s = _tokens(sentence)
    if not s:
        return True
    corpus_tokens = set()
    for p in passages:
        corpus_tokens |= _tokens(p.get("content", ""))
    overlap = len(s & corpus_tokens) / len(s)
    return overlap >= min_overlap
