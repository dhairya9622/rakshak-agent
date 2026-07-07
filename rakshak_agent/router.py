"""
Model routing, LLM cascade execution and cost accounting.

Tiers, cheapest first. A request climbs only as far as it must:

  TIER_RESPONSE_CACHE   exact repeat            zero cost
  TIER_SEMANTIC_CACHE   paraphrase repeat       zero cost
  TIER_DETERMINISTIC    offline agent           zero cost
  TIER_CHEAP_LLM        MEDIUM -> cheap model   low cost
  TIER_CAPABLE_LLM      HARD   -> cascade       higher cost, only if cheap abstains
  TIER_REFUSED          out-of-scope            zero cost (never calls a model)

CostMeter aggregates usage so tests (and a dashboard) can assert the LLM-call
rate and spend stay low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import classifier, context
from .cache import PromptPrefix
from .llm import ModelCascade

TIER_RESPONSE_CACHE = "response_cache"
TIER_SEMANTIC_CACHE = "semantic_cache"
TIER_DETERMINISTIC = "deterministic"
TIER_CHEAP_LLM = "cheap_llm"
TIER_CAPABLE_LLM = "capable_llm"
TIER_REFUSED = "refused"

_ZERO_COST_TIERS = {TIER_RESPONSE_CACHE, TIER_SEMANTIC_CACHE,
                    TIER_DETERMINISTIC, TIER_REFUSED}


@dataclass
class CostMeter:
    requests: int = 0
    deterministic: int = 0
    response_cache_hits: int = 0
    semantic_cache_hits: int = 0
    cheap_calls: int = 0
    capable_calls: int = 0
    refusals: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    total_usd: float = 0.0

    def record(self, tier: str, resp=None):
        self.requests += 1
        if tier == TIER_DETERMINISTIC:
            self.deterministic += 1
        elif tier == TIER_RESPONSE_CACHE:
            self.response_cache_hits += 1
        elif tier == TIER_SEMANTIC_CACHE:
            self.semantic_cache_hits += 1
        elif tier == TIER_REFUSED:
            self.refusals += 1

    def record_llm(self, attempts):
        for a in attempts:
            self.tokens_in += a.tokens_in
            self.tokens_out += a.tokens_out
            self.total_usd += a.cost_usd
            if a.tokens_in or a.tokens_out or a.cost_usd or a.meta.get("error"):
                if getattr(a, "tier_used", "cheap") == "capable":
                    self.capable_calls += 1
                else:
                    self.cheap_calls += 1

    @property
    def llm_calls(self) -> int:
        return self.cheap_calls + self.capable_calls

    def summary(self) -> Dict:
        zero_cost = (self.deterministic + self.response_cache_hits
                     + self.semantic_cache_hits + self.refusals)
        rate = (self.llm_calls / self.requests) if self.requests else 0.0
        return {
            "requests": self.requests,
            "zero_cost_answers": zero_cost,
            "zero_cost_fraction": round(zero_cost / self.requests, 3) if self.requests else 0.0,
            "response_cache_hits": self.response_cache_hits,
            "semantic_cache_hits": self.semantic_cache_hits,
            "deterministic": self.deterministic,
            "refusals": self.refusals,
            "cheap_llm_calls": self.cheap_calls,
            "capable_llm_calls": self.capable_calls,
            "llm_call_rate": round(rate, 3),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_usd": round(self.total_usd, 6),
        }


@dataclass
class Escalation:
    answered: bool
    text: str = ""
    sources: List[Dict] = field(default_factory=list)
    tier: str = TIER_DETERMINISTIC
    model: Optional[str] = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    prompt_cache_hit: bool = False
    attempts: list = field(default_factory=list)
    system_prompt: str = ""
    user_prompt: str = ""


class Router:
    """Builds minimal context and runs the cheap->capable cascade, respecting
    the difficulty ceiling (MEDIUM never touches the capable model)."""

    def __init__(self, kb, index, embed_index, prompt_prefix: PromptPrefix,
                 cheap_client=None, capable_client=None,
                 context_budget: int = 380):
        self.kb = kb
        self.index = index
        self.embed_index = embed_index
        self.prefix = prompt_prefix
        self.cheap = cheap_client
        self.capable = capable_client
        self.context_budget = context_budget

    def _clients_for(self, level: str):
        if level == classifier.MEDIUM:
            chain = [self.cheap]                     # cheap only
        elif level == classifier.HARD:
            chain = [self.cheap, self.capable]       # cheap first, then capable
        else:
            chain = []
        chain = [c for c in chain if c is not None]
        return chain

    def escalate(self, question: str, difficulty) -> Escalation:
        clients = self._clients_for(difficulty.level)
        if not clients:
            return Escalation(answered=False)  # no model available at this tier

        ctx, sources = context.select_context(
            question, self.kb, self.index, self.embed_index, self.context_budget)
        if not ctx.strip():
            return Escalation(answered=False)

        user = context.build_user_prompt(question, ctx)
        cascade = ModelCascade(clients)
        cached_prefix = self.prefix.token_size()  # stable system prompt -> prompt cache
        resp, attempts = cascade.run(self.prefix.system, user, cached_prefix_tokens=cached_prefix)

        # Tag which tier each attempt belonged to (for the cost meter).
        for a, client in zip(attempts, clients):
            a.tier_used = getattr(client, "tier", "cheap")

        if resp is None:
            return Escalation(answered=False, attempts=attempts, sources=sources)

        winning_tier = TIER_CAPABLE_LLM if getattr(
            resp, "model", "") == getattr(self.capable, "name", None) else TIER_CHEAP_LLM
        # More robust: map by the client that produced resp.
        for a, client in zip(attempts, clients):
            if a is resp:
                winning_tier = (TIER_CAPABLE_LLM if getattr(client, "tier", "cheap") == "capable"
                                else TIER_CHEAP_LLM)
        return Escalation(
            answered=True, text=resp.text, sources=sources, tier=winning_tier,
            model=resp.model, cost_usd=sum(a.cost_usd for a in attempts),
            tokens_in=sum(a.tokens_in for a in attempts),
            tokens_out=sum(a.tokens_out for a in attempts),
            prompt_cache_hit=resp.prompt_cache_hit, attempts=attempts,
            system_prompt=self.prefix.system, user_prompt=user)
