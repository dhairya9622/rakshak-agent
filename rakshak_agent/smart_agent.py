"""
SmartAgent - the cost-optimised orchestrator.

It wraps the offline deterministic Agent and adds the escalation architecture:

  request
    -> exact response cache        (hit? return, zero cost)
    -> run deterministic agent      (always; it is free and often sufficient)
    -> semantic (paraphrase) cache  (hit? return, zero cost)
    -> classify difficulty
        EASY          -> return the deterministic answer          (zero cost)
        OUT_OF_SCOPE  -> return refusal                           (zero cost, no model)
        MEDIUM        -> cheap model over minimal context (or det fallback)
        HARD          -> cheap->capable cascade (or det fallback)
    -> cache the result (response + semantic)

With no LLM configured it is a pure offline agent: every path that would
escalate degrades gracefully to the best deterministic answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import classifier, normalize
from .cache import PromptPrefix, ResponseCache, SemanticCache
from .embeddings import EmbeddingIndex, LocalEmbedder
from .engine import Agent
from .router import (CostMeter, Router, TIER_CAPABLE_LLM, TIER_CHEAP_LLM,
                     TIER_DETERMINISTIC, TIER_REFUSED, TIER_RESPONSE_CACHE,
                     TIER_SEMANTIC_CACHE)


@dataclass
class SmartAnswer:
    text: str
    intent: str
    tier: str
    difficulty: str
    in_scope: bool = True
    confidence: float = 0.0
    llm_used: bool = False
    cached: Optional[str] = None          # None | 'response' | 'semantic'
    model: Optional[str] = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    prompt_cache_hit: bool = False
    topic: Optional[str] = None            # resolved subject; echo back as context
    sources: List[Dict] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["cost_usd"] = round(self.cost_usd, 6)
        return d

    # payload persisted in caches (answer content only, not cost/tier)
    def _payload(self) -> Dict:
        return {"text": self.text, "intent": self.intent, "in_scope": self.in_scope,
                "confidence": self.confidence, "sources": self.sources,
                "data": self.data, "model": self.model, "llm_used": self.llm_used}


class SmartAgent:
    def __init__(self, kb, cheap_llm=None, capable_llm=None,
                 embed_dim: int = 512, semantic_threshold: float = 0.82,
                 context_budget: int = 380, teacher_logger=None, clock=None):
        self.agent = Agent(kb, clock=clock)
        self.teacher_logger = teacher_logger   # captures capable-model outputs
        self.kb = kb
        self.index = self.agent.index

        self.embedder = LocalEmbedder(dim=embed_dim)
        self.embed_index = EmbeddingIndex(self.embedder, kb.chunks)

        self.response_cache = ResponseCache()
        self.semantic_cache = SemanticCache(self.embedder, threshold=semantic_threshold)
        self.prefix = PromptPrefix(kb.reports)
        self.router = Router(kb, self.index, self.embed_index, self.prefix,
                             cheap_client=cheap_llm, capable_client=capable_llm,
                             context_budget=context_budget)
        self.meter = CostMeter()

    @classmethod
    def load(cls, knowledge_dir: str = "knowledge", cheap_llm=None,
             capable_llm=None, clock=None, **kw) -> "SmartAgent":
        from .knowledge import KnowledgeBase
        return cls(KnowledgeBase.load(knowledge_dir),
                   cheap_llm=cheap_llm, capable_llm=capable_llm, clock=clock, **kw)

    # ------------------------------------------------------------------ #

    def _in_domain(self, question: str) -> float:
        raw = normalize.content_tokens(question)
        alpha = [t for t in raw if any(c.isalpha() for c in t)]
        return self.index.known_fraction(alpha) if alpha else 0.0

    def _from_cache(self, payload: Dict, tier: str, cached_kind: str,
                    difficulty: str) -> SmartAnswer:
        return SmartAnswer(
            text=payload["text"], intent=payload.get("intent", "cached"),
            tier=tier, difficulty=difficulty, in_scope=payload.get("in_scope", True),
            confidence=payload.get("confidence", 0.0), llm_used=False,
            cached=cached_kind, model=payload.get("model"),
            cost_usd=0.0, topic=(payload.get("data") or {}).get("topic"),
            sources=payload.get("sources", []), data=payload.get("data", {}))

    def _from_det(self, det, tier: str, difficulty: str) -> SmartAnswer:
        return SmartAnswer(
            text=det.text, intent=det.intent, tier=tier, difficulty=difficulty,
            in_scope=det.in_scope, confidence=det.confidence, llm_used=False,
            cost_usd=0.0, topic=(det.data or {}).get("topic"),
            sources=det.sources, data=det.data)

    # ------------------------------------------------------------------ #

    def ask(self, question: str, context: dict = None) -> SmartAnswer:
        """context (optional): {"last_entity": "<name>"} lets a follow-up like
        'advise on this' resolve to the prior turn's topic. Context-dependent
        follow-ups bypass the caches so answers never bleed across topics."""
        q = (question or "").strip()
        if not q:
            self.meter.record(TIER_DETERMINISTIC)
            return SmartAnswer("Please ask a question about the Rakshak reports.",
                               "empty", TIER_DETERMINISTIC, classifier.EASY)

        from .engine import is_anaphoric
        use_cache = not (context and is_anaphoric(q))

        # 1) exact response cache
        if use_cache:
            hit = self.response_cache.get(q)
            if hit is not None:
                self.meter.record(TIER_RESPONSE_CACHE)
                return self._from_cache(hit, TIER_RESPONSE_CACHE, "response", classifier.EASY)

        # deterministic agent (free) + routing signals
        det = self.agent.ask(q, context=context)
        in_domain = self._in_domain(q)
        modules = normalize.detect_modules(q)
        entity = self.index.find_entity(q)
        if entity is None and context and context.get("last_entity") and is_anaphoric(q):
            entity = self.index.kb.entity_by_name.get(context["last_entity"])
        topic = SemanticCache.topic_key(modules, entity["name"] if entity else None)

        # 2) semantic (paraphrase) cache, topic-guarded
        if use_cache:
            sem = self.semantic_cache.get(q, topic)
            if sem is not None:
                self.meter.record(TIER_SEMANTIC_CACHE)
                return self._from_cache(sem, TIER_SEMANTIC_CACHE, "semantic", classifier.EASY)

        # 3) classify difficulty
        diff = classifier.classify(q, det, in_domain)

        # 4) route
        if diff.level == classifier.EASY:
            ans = self._from_det(det, TIER_DETERMINISTIC, diff.level)
            self.meter.record(TIER_DETERMINISTIC)
        elif diff.level == classifier.OUT_OF_SCOPE:
            ans = self._from_det(det, TIER_REFUSED, diff.level)
            self.meter.record(TIER_REFUSED)
        else:
            ans = self._route_llm(q, det, diff)

        if ans.topic is None and entity:
            ans.topic = entity["name"]

        # 5) cache the result for future exact/paraphrase repeats
        if use_cache:
            self.response_cache.put(q, ans._payload())
            self.semantic_cache.put(q, topic, ans._payload())
        return ans

    def _route_llm(self, q, det, diff) -> SmartAnswer:
        esc = self.router.escalate(q, diff)
        if esc.answered:
            self.meter.requests += 1
            self.meter.record_llm(esc.attempts)
            if self.teacher_logger is not None:
                self.teacher_logger.record(q, esc.system_prompt, esc.user_prompt,
                                           esc.text, esc.model, esc.tier, esc.sources)
            return SmartAnswer(
                text=esc.text, intent="llm", tier=esc.tier, difficulty=diff.level,
                in_scope=True, confidence=1.0, llm_used=True, model=esc.model,
                cost_usd=esc.cost_usd, tokens_in=esc.tokens_in,
                tokens_out=esc.tokens_out, prompt_cache_hit=esc.prompt_cache_hit,
                sources=esc.sources, data={"escalated": True})

        # Cascade unavailable or everyone abstained -> graceful offline fallback.
        if esc.attempts:  # a model was actually called but abstained
            self.meter.requests += 1
            self.meter.record_llm(esc.attempts)
        if diff.level == classifier.HARD and not det.in_scope:
            ans = self._from_det(det, TIER_REFUSED, diff.level)
            self.meter.record(TIER_REFUSED)
        else:
            ans = self._from_det(det, TIER_DETERMINISTIC, diff.level)
            if not esc.attempts:
                self.meter.record(TIER_DETERMINISTIC)
        return ans

    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        s = self.meter.summary()
        s["response_cache"] = {"hits": self.response_cache.hits,
                               "misses": self.response_cache.misses}
        s["semantic_cache"] = {"hits": self.semantic_cache.hits,
                               "misses": self.semantic_cache.misses}
        return s
