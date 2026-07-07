"""
Offline embedding / indexing pipeline.

A dependency-free, fully deterministic vectoriser used for:
  * semantic caching  (detect paraphrases of previously answered questions)
  * semantic retrieval (complement BM25 when selecting minimal LLM context)

There is NO external model and NO network. Vectors are hashed token/bigram
bags (feature hashing), so the same text always produces the same vector -
which keeps the whole cost/routing layer replayable. Swap LocalEmbedder for a
real embedding client later without touching callers (same .embed() contract).
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Dict, List, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


class LocalEmbedder:
    """Deterministic sparse-hashed embedding (unigrams + bigrams)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> List[str]:
        toks = _TOKEN.findall((text or "").lower())
        feats = list(toks)
        for i in range(len(toks) - 1):
            feats.append(toks[i] + "_" + toks[i + 1])
        return feats

    def _bucket(self, feature: str) -> int:
        # crc32 is stable across processes/runs (unlike Python's salted hash()).
        return zlib.crc32(feature.encode("utf-8")) % self.dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for f in self._features(text):
            vec[self._bucket(f)] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class EmbeddingIndex:
    """In-memory semantic index over the KB chunks (built at load time)."""

    def __init__(self, embedder: LocalEmbedder, chunks: List[Dict]):
        self.embedder = embedder
        self.ids: List[str] = []
        self.vectors: List[List[float]] = []
        self.by_id: Dict[str, Dict] = {}
        for c in chunks:
            self.ids.append(c["chunk_id"])
            self.by_id[c["chunk_id"]] = c
            self.vectors.append(embedder.embed(
                (c.get("section", "") or "") + " " + (c.get("text", "") or "")))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        qv = self.embedder.embed(query)
        scored = []
        for cid, v in zip(self.ids, self.vectors):
            scored.append((cosine(qv, v), cid))
        # score desc, id asc -> deterministic
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(s, self.by_id[cid]) for s, cid in scored[:top_k] if s > 0]
