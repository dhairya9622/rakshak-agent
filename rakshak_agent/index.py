"""
Deterministic retrieval index.

Pure-Python, offline, no ML libraries. Provides:
  * a BM25-lite ranker over the section chunks (free-text / explanatory Qs)
  * scored lookups over facts, verdicts and entities (precise Qs)

Determinism guarantees:
  * scoring is a fixed arithmetic function of token counts
  * every ranking breaks ties on a stable key (id), never on hash/order/time
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .knowledge import KnowledgeBase
from . import normalize


# BM25 parameters (standard, fixed).
_K1 = 1.5
_B = 0.75


class Index:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._build_chunk_index()
        self._build_vocab()

    # ------------------------------------------------------------------ #
    # Chunk BM25 index
    # ------------------------------------------------------------------ #

    def _build_chunk_index(self) -> None:
        self.chunk_tokens: List[List[str]] = []
        self.chunk_tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        total_len = 0
        for c in self.kb.chunks:
            toks = normalize.content_tokens(
                (c.get("section", "") or "") + " " + (c.get("text", "") or ""))
            self.chunk_tokens.append(toks)
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.chunk_tf.append(tf)
            total_len += len(toks)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.n_docs = max(1, len(self.kb.chunks))
        self.avg_len = (total_len / self.n_docs) if self.n_docs else 1.0

    def _idf(self, term: str) -> float:
        # BM25 idf with +1 smoothing (never negative).
        n_t = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n_t + 0.5) / (n_t + 0.5))

    def weights(self, terms: List[str]) -> Dict[str, float]:
        """IDF weight per term, for rare-word-aware sentence selection."""
        return {t: self._idf(t) for t in set(terms)}

    def _build_vocab(self) -> None:
        """Union vocabulary across all knowledge, for out-of-scope detection."""
        vocab = set(self.df.keys())
        for e in self.kb.entities:
            for t in normalize.content_tokens(e["name"] + " " + " ".join(e.get("aliases", []))):
                vocab.add(t)
        for f in self.kb.facts:
            for t in f.get("tags", []):
                vocab.add(normalize._light_stem(t))
            for t in normalize.content_tokens(str(f.get("subject") or "")):
                vocab.add(t)
        for r in self.kb.reports:
            for t in normalize.content_tokens(r.get("report_id", "") + " " + (r.get("module_label") or "")):
                vocab.add(t)
        self.vocab = vocab

    def known_fraction(self, terms: List[str]) -> float:
        """Fraction of query terms that exist anywhere in the knowledge."""
        if not terms:
            return 0.0
        known = sum(1 for t in terms if t in self.vocab)
        return known / len(terms)

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #

    def rank_chunks(self, terms: List[str], modules: List[str],
                    top_k: int = 6) -> List[Tuple[float, Dict]]:
        scored: List[Tuple[float, str, Dict]] = []
        qset = list(dict.fromkeys(terms))
        for i, c in enumerate(self.kb.chunks):
            tf = self.chunk_tf[i]
            dl = len(self.chunk_tokens[i]) or 1
            score = 0.0
            for t in qset:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = self._idf(t)
                denom = f + _K1 * (1 - _B + _B * dl / self.avg_len)
                score += idf * (f * (_K1 + 1)) / denom
            if modules and c["module"] in modules:
                score *= 1.35  # bias toward the targeted report
            if score > 0:
                scored.append((score, c["chunk_id"], c))
        # sort by score desc, then chunk_id asc  -> deterministic
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(s, c) for s, _, c in scored[:top_k]]

    def rank_facts(self, terms: List[str], modules: List[str], kinds=None,
                   top_k: int = 6) -> List[Tuple[float, Dict]]:
        qset = set(terms)
        scored: List[Tuple[float, str, Dict]] = []
        for f in self.kb.facts:
            if kinds and f["kind"] not in kinds:
                continue
            tagset = {normalize._light_stem(t) for t in f.get("tags", [])}
            subj = set(normalize.content_tokens(str(f.get("subject") or "")))
            ctx = set(normalize.content_tokens(str(f.get("context") or "")))
            overlap = len(qset & tagset) * 2.0 + len(qset & subj) * 3.0 + len(qset & ctx) * 1.0
            if overlap <= 0:
                continue
            if modules and f["module"] in modules:
                overlap *= 1.4
            # Readability: prefer clean, short contexts (P1 tiles / glance) over
            # long interleaved P2 ledger run-ons when the signal is comparable.
            clen = len(str(f.get("context") or ""))
            readability = 200.0 / clen if clen > 200 else 1.0
            overlap *= (0.7 + 0.3 * readability)
            scored.append((overlap, f["fact_id"], f))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(s, f) for s, _, f in scored[:top_k]]

    def rank_verdicts(self, terms: List[str], modules: List[str],
                      entity: str = None, top_k: int = 6) -> List[Tuple[float, Dict]]:
        qset = set(terms)
        scored: List[Tuple[float, str, Dict]] = []
        for v in self.kb.verdicts:
            hay = set(normalize.content_tokens(
                " ".join(filter(None, [
                    v.get("item"), v.get("party"), v.get("verdict_alias"),
                    v.get("verdict_class"), v.get("verdict_meaning"), v.get("text"),
                ])) + " " + " ".join(v.get("citations", []))))
            overlap = len(qset & hay)
            if entity and v.get("party") == entity:
                overlap += 5
            if modules and v["module"] in modules:
                overlap = overlap * 1.4 + 0.5
            if overlap <= 0:
                continue
            scored.append((float(overlap), v["verdict_id"], v))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(s, v) for s, _, v in scored[:top_k]]

    def find_entity(self, text: str):
        """Return the entity dict whose name/alias appears in the question."""
        folded = normalize.fold(text)
        best = None
        best_len = 0
        for e in self.kb.entities:
            names = [e["name"]] + e.get("aliases", [])
            for nm in names:
                if normalize.fold(nm) in folded and len(nm) > best_len:
                    best, best_len = e, len(nm)
        return best
