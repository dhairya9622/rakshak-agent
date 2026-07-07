"""
Minimal-context selection for the LLM (the "R" in Agentic RAG).

When (and only when) a request escalates to a model, we send the SMALLEST
relevant slice of the deterministic knowledge - never whole reports. Context is
a hybrid of:
  * BM25 lexical retrieval (exact terms, citations)
  * embedding retrieval (paraphrase / semantic)
  * the top structured facts & verdicts already extracted

Everything is trimmed to a small token budget. Fewer tokens = lower cost and
tighter grounding (less room to hallucinate).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from . import normalize
from .llm import estimate_tokens


def _merge_chunks(bm25: List[Tuple[float, Dict]],
                  semantic: List[Tuple[float, Dict]]) -> List[Dict]:
    """Reciprocal-rank fusion of the two retrievers (deterministic)."""
    scores: Dict[str, float] = {}
    keep: Dict[str, Dict] = {}
    for rank, (_, c) in enumerate(bm25):
        scores[c["chunk_id"]] = scores.get(c["chunk_id"], 0.0) + 1.0 / (rank + 1)
        keep[c["chunk_id"]] = c
    for rank, (_, c) in enumerate(semantic):
        scores[c["chunk_id"]] = scores.get(c["chunk_id"], 0.0) + 1.0 / (rank + 1)
        keep[c["chunk_id"]] = c
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [keep[cid] for cid, _ in order]


def select_context(question: str, kb, index, embed_index,
                   token_budget: int = 380,
                   max_chunks: int = 4) -> Tuple[str, List[Dict]]:
    terms = normalize.query_terms(question)
    modules = normalize.detect_modules(question)

    bm25 = index.rank_chunks(terms, modules, top_k=6)
    semantic = embed_index.search(question, top_k=6)
    fused = _merge_chunks(bm25, semantic)

    # Bias to the targeted report if the question named one.
    if modules:
        fused.sort(key=lambda c: (0 if c["module"] in modules else 1,))

    pieces: List[str] = []
    sources: List[Dict] = []
    used = 0
    for c in fused[:max_chunks]:
        snippet = c["text"]
        if len(snippet) > 500:
            snippet = snippet[:500].rstrip() + " …"
        line = "[%s p%d] %s" % (c["report_id"], c["page"], snippet)
        t = estimate_tokens(line)
        if used + t > token_budget and pieces:
            break
        pieces.append(line)
        sources.append({"report_id": c["report_id"], "page": c["page"]})
        used += t

    # Add the single most relevant structured verdict/fact if room remains.
    vs = index.rank_verdicts(terms, modules, top_k=1)
    if vs and used < token_budget:
        v = vs[0][1]
        line = "[%s verdict] %s -> %s (%s)%s" % (
            v["report_id"], v.get("item") or v.get("party"),
            v.get("verdict_alias"), v.get("verdict_class"),
            (" · basis: " + " · ".join(v.get("citations", [])[:3])) if v.get("citations") else "")
        if estimate_tokens(line) + used <= token_budget:
            pieces.append(line)
            sources.append({"report_id": v["report_id"], "page": 2})

    context = "\n".join(pieces)
    return context, sources


def build_user_prompt(question: str, context: str) -> str:
    return ("CONTEXT:\n%s\n\nQUESTION: %s\n\n"
            "Answer from CONTEXT only, cite report id(s), or reply "
            "INSUFFICIENT_CONTEXT." % (context, question))
