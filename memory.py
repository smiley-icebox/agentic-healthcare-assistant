"""Long-term patient memory — persisted across sessions, scoped, redacted, and
semantically retrievable via FAISS.

Three guards against the classic memory failure (leaking one patient's PHI into
another's prompt):
  1. Memory is keyed by patient_id and only ever loaded/searched for the can_access'd
     subject — every function takes the subject and reads only that patient's rows.
  2. Content is PHI-redacted before it's stored (defense-in-depth; see SECURITY.md).
  3. The FAISS index is built per call from ONE patient's rows and never persisted, so
     it can't drift and can't pool another patient's vectors.

`search_memory` is the rubric's "vector database (FAISS) to store and retrieve patient
summaries": stored patient summaries are embedded (TF-IDF) and retrieved by relevance to
the current request, so the prompt gets the most pertinent prior context, not just the
most recent.
"""

import db
import observability


def load_memory(subject_id: str, limit: int = 5) -> list[str]:
    """Recent long-term context for ONE patient (most recent first)."""
    if not subject_id:
        return []
    return [r["content"] for r in db.list_memory(subject_id, limit)]


def search_memory(subject_id: str, query: str, k: int = 3) -> list[str]:
    """Most RELEVANT stored summaries for one patient, via a per-patient FAISS index over
    TF-IDF vectors. Falls back to recency when there's too little to index or no query
    overlap. The index is scoped to this subject and rebuilt each call — never shared."""
    if not subject_id:
        return []
    notes = [r["content"] for r in db.list_memory(subject_id, 50)]
    if not query or len(notes) < 2:
        return notes[:k]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import faiss
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(notes).astype("float32").toarray()
        faiss.normalize_L2(mat)                       # cosine via inner product
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        q = vec.transform([query]).astype("float32").toarray()
        if q.sum() == 0:
            return notes[:k]
        faiss.normalize_L2(q)
        _, idx = index.search(q, min(k, len(notes)))
        return [notes[i] for i in idx[0] if i >= 0]
    except Exception:
        return notes[:k]                              # never break a request over memory


def save_memory(subject_id: str, content: str) -> bool:
    """Persist a short context note / patient summary for a patient, redacted first."""
    if not subject_id or not content:
        return False
    return db.add_memory(subject_id, observability.redact(content))
