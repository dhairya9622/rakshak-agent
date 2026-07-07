"""
Caching layer - three kinds, all aimed at avoiding paid LLM calls.

1. ResponseCache   exact (normalised) question -> stored answer. Zero cost reuse
                   of anything answered before (deterministic or LLM).
2. SemanticCache   paraphrase reuse: embed the question and reuse a prior answer
                   when cosine >= threshold AND the topic guard matches (same
                   module/entity) so 'ITC due date' never returns 'TDS due date'.
3. PromptPrefix    the STABLE system prompt (scope + rules). Kept byte-identical
                   every request so the provider's context/prompt cache is hit;
                   the small, varying retrieved context goes in the user turn.

All in-memory and deterministic. Persisting/seeding is a drop-in later.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .embeddings import LocalEmbedder, cosine


def normalise(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


class ResponseCache:
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, question: str) -> Optional[dict]:
        key = normalise(question)
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        return None

    def put(self, question: str, payload: dict) -> None:
        key = normalise(question)
        self._store[key] = payload
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)


class SemanticCache:
    def __init__(self, embedder: LocalEmbedder, threshold: float = 0.86,
                 capacity: int = 2048):
        self.embedder = embedder
        self.threshold = threshold
        self.capacity = capacity
        # list of (vector, topic_key, payload)
        self._entries: List[Tuple[List[float], str, dict]] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def topic_key(modules: List[str], entity: Optional[str]) -> str:
        # Topic guard: paraphrases only collide within the same report(s)/party.
        return "|".join(sorted(modules or [])) + "#" + (entity or "")

    def get(self, question: str, topic: str) -> Optional[dict]:
        qv = self.embedder.embed(question)
        best, best_sim = None, self.threshold
        for vec, tkey, payload in self._entries:
            if tkey != topic:
                continue
            sim = cosine(qv, vec)
            if sim >= best_sim:
                best_sim, best = sim, payload
        if best is not None:
            self.hits += 1
            return best
        self.misses += 1
        return None

    def put(self, question: str, topic: str, payload: dict) -> None:
        self._entries.append((self.embedder.embed(question), topic, payload))
        if len(self._entries) > self.capacity:
            self._entries.pop(0)


class PromptPrefix:
    """Builds the stable system prompt once. Same bytes every call => provider
    prompt-cache friendly. Also exposes its token size for cost accounting."""

    def __init__(self, reports: List[dict]):
        lines = [
            "You are Rakshak Assistant, a careful assistant for a Chartered "
            "Accountant. You answer ONLY from the CONTEXT provided in the user "
            "message, which is extracted from deterministic compliance reports.",
            "Rules:",
            "- Use only facts present in CONTEXT. Never invent numbers, dates, "
            "section citations, or verdicts.",
            "- Cite the report id(s) you used.",
            "- Be concise and precise; prefer the exact figures/verdicts shown.",
            "- If the CONTEXT does not contain the answer, reply with exactly: "
            "INSUFFICIENT_CONTEXT",
            "Known reports (scope):",
        ]
        for r in reports:
            lines.append("- %s: %s%s" % (
                r["report_id"], r.get("module_label", r.get("module", "")),
                (" (" + r["period"] + ")") if r.get("period") else ""))
        self.system = "\n".join(lines)

    def token_size(self) -> int:
        from .llm import estimate_tokens
        return estimate_tokens(self.system)
