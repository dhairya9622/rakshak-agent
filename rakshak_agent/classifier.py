"""
Question difficulty classifier.

Runs on EVERY request, before any LLM call, using only cheap deterministic
signals already produced by the offline agent. Its output decides how far up
the cost ladder a request is allowed to climb.

Difficulty -> intended tier:
  EASY          the deterministic agent answered confidently        -> offline (zero cost)
  MEDIUM        in-scope but low confidence / fuzzy extractive       -> cache, else cheap LLM
  HARD          in-domain but the agent could not answer well /
                ambiguous / multi-hop reasoning                      -> cheap->capable cascade
  OUT_OF_SCOPE  question is not about the reports at all             -> refuse (NEVER call LLM)

The key cost lever: genuinely external questions are OUT_OF_SCOPE, so they are
refused for free and never reach a paid model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import intents

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"
OUT_OF_SCOPE = "out_of_scope"

# Confidence a *structured* deterministic answer needs to count as EASY.
_STRUCT_EASY = 1.0
# Confidence an *extractive* (explain/define) answer needs to count as EASY.
_EXTRACTIVE_EASY = 4.0
# Below this in-domain fraction, an unanswered question is treated as external.
_IN_DOMAIN = 0.34

_STRUCTURED = {intents.IDENTITY, intents.AMOUNT, intents.COUNT,
               intents.LIST, intents.VERDICT, intents.ADVICE}

# Cues that a question wants reasoning/synthesis the symbolic layer can't do
# well on its own (comparisons, causal/hypothetical, cross-report narrative).
_REASONING_RE = re.compile(
    r"\b(compare|difference between|versus|vs\.?|why|how come|what if|"
    r"should (i|we)|recommend|implication|trade[- ]?off|walk me through|"
    r"summar(y|ise|ize)|explain the relationship|across (all|the) reports|"
    r"over the (two years|timeline)|what does this mean for|what it means|"
    r"in plain (english|terms)|connects? to|how .* relate|"
    r"explain (how|why)|in simple terms)\b")


@dataclass
class Difficulty:
    level: str
    reason: str
    in_domain: float          # 0..1 fraction of alphabetic terms known to KB
    reasoning: bool           # question asks for synthesis/reasoning


def classify(question: str, det_answer, in_domain: float) -> Difficulty:
    reasoning = bool(_REASONING_RE.search(question.lower()))

    if det_answer is None or not det_answer.in_scope:
        if in_domain >= _IN_DOMAIN:
            return Difficulty(HARD, "in-domain but agent could not answer",
                              in_domain, reasoning)
        return Difficulty(OUT_OF_SCOPE, "not about the reports", in_domain, reasoning)

    intent = det_answer.intent
    conf = det_answer.confidence

    # The advisory engine IS the CA reasoning; never pay an LLM to rephrase it.
    if intent == intents.ADVICE and conf >= 1.0:
        return Difficulty(EASY, "deterministic advisory", in_domain, False)

    # A reasoning/synthesis request is at least MEDIUM even if the agent found
    # something, because the offline answer is usually a raw extract.
    if reasoning:
        # confident structured facts still answer many "why" cases well enough
        if intent in _STRUCTURED and conf >= _STRUCT_EASY:
            return Difficulty(MEDIUM, "reasoning cue over a solid fact", in_domain, True)
        return Difficulty(HARD, "reasoning/synthesis requested", in_domain, True)

    if intent in _STRUCTURED and conf >= _STRUCT_EASY:
        return Difficulty(EASY, "confident structured answer", in_domain, False)
    if intent in (intents.EXPLAIN, intents.DEFINE) and conf >= _EXTRACTIVE_EASY:
        return Difficulty(EASY, "strong extractive answer", in_domain, False)
    if intent in ("capabilities", "empty"):
        return Difficulty(EASY, "meta", in_domain, False)

    return Difficulty(MEDIUM, "in-scope but weak confidence", in_domain, reasoning)
