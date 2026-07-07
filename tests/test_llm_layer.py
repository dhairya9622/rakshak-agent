#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the cost-optimised LLM escalation layer.

Focus (per the deliverable):
  * routing accuracy            - right tier for easy / medium / hard / oos
  * zero-cost guarantees        - easy & out-of-scope never call a model
  * cache hit rate              - exact + semantic (paraphrase) reuse
  * cost reduction              - low llm-call rate, prompt-cache discount
  * fallback behaviour          - LLM error / abstain -> graceful offline answer
  * answer quality              - offline answer == deterministic; LLM grounded

Runs with the stdlib only:  python -m unittest discover -s tests
Uses deterministic mock LLM clients, so the whole suite is reproducible.
"""

import io
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from rakshak_agent import (SmartAgent, MockLLMClient, ScriptedLLMClient, ABSTAIN)
from rakshak_agent import classifier
from rakshak_agent.llm import LLMClient

KNOWLEDGE_DIR = os.path.join(_ROOT, "knowledge")


def cheap_answering(name="deepseek-cheap"):
    return MockLLMClient(name, tier="cheap",
                         handler=lambda s, u: "Cheap grounded answer. [ITC-2627-MAY-0049]",
                         price_in_per_m=0.27, price_out_per_m=1.10,
                         price_in_cached_per_m=0.07)


def cheap_abstaining(name="deepseek-cheap"):
    return MockLLMClient(name, tier="cheap", handler=lambda s, u: ABSTAIN,
                         price_in_per_m=0.27, price_out_per_m=1.10)


def capable_answering(name="capable-model"):
    return MockLLMClient(name, tier="capable",
                         handler=lambda s, u: "Capable synthesised answer. [ANN-2425-FY-0012]",
                         price_in_per_m=3.0, price_out_per_m=15.0)


class ExplodingClient(LLMClient):
    def __init__(self, name="broken", tier="cheap"):
        super().__init__(name, tier, price_in_per_m=0.27, price_out_per_m=1.10)

    def _complete(self, system, user):
        raise RuntimeError("network down")


EASY_Q = "What is the GSTIN?"
OOS_Q = "What is the capital of France?"
HARD_Q = "Why does the spillover credit matter for GSTR-9 mismatch notices?"


# --------------------------------------------------------------------------- #
# Offline-only mode: the foundation must stand alone at zero cost
# --------------------------------------------------------------------------- #

class TestOfflineOnly(unittest.TestCase):

    def setUp(self):
        self.sa = SmartAgent.load(KNOWLEDGE_DIR)  # no LLM

    def test_easy_is_zero_cost_deterministic(self):
        a = self.sa.ask(EASY_Q)
        self.assertFalse(a.llm_used)
        self.assertEqual(a.cost_usd, 0.0)
        self.assertIn("27XXXXX1234X1Z5", a.text)

    def test_out_of_scope_refused_no_model(self):
        a = self.sa.ask(OOS_Q)
        self.assertFalse(a.in_scope)
        self.assertFalse(a.llm_used)
        self.assertEqual(a.cost_usd, 0.0)

    def test_no_llm_call_rate_is_zero(self):
        for q in [EASY_Q, OOS_Q, HARD_Q, "List the vendors", "How much RCM?"]:
            self.sa.ask(q)
        s = self.sa.stats()
        self.assertEqual(s["llm_call_rate"], 0.0)
        self.assertEqual(s["total_usd"], 0.0)

    def test_hard_falls_back_gracefully_without_llm(self):
        a = self.sa.ask(HARD_Q)
        self.assertFalse(a.llm_used)
        self.assertTrue(a.text.strip())  # still returns the best offline answer


# --------------------------------------------------------------------------- #
# Classification / routing accuracy
# --------------------------------------------------------------------------- #

class TestRouting(unittest.TestCase):

    def fresh(self, cheap=None, capable=None):
        return SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=cheap, capable_llm=capable)

    def test_easy_never_calls_model(self):
        cheap = cheap_answering()
        sa = self.fresh(cheap=cheap)
        a = sa.ask(EASY_Q)
        self.assertEqual(a.tier, "deterministic")
        self.assertEqual(cheap.calls, 0)

    def test_out_of_scope_never_calls_model(self):
        cheap, capable = cheap_answering(), capable_answering()
        sa = self.fresh(cheap=cheap, capable=capable)
        a = sa.ask(OOS_Q)
        self.assertEqual(a.tier, "refused")
        self.assertEqual(cheap.calls, 0)
        self.assertEqual(capable.calls, 0)

    def test_hard_uses_cheap_first_and_stops(self):
        cheap, capable = cheap_answering(), capable_answering()
        sa = self.fresh(cheap=cheap, capable=capable)
        a = sa.ask(HARD_Q)
        self.assertTrue(a.llm_used)
        self.assertEqual(a.tier, "cheap_llm")
        self.assertEqual(cheap.calls, 1)
        self.assertEqual(capable.calls, 0)   # cheap answered -> no escalation

    def test_hard_escalates_to_capable_when_cheap_abstains(self):
        cheap, capable = cheap_abstaining(), capable_answering()
        sa = self.fresh(cheap=cheap, capable=capable)
        a = sa.ask(HARD_Q)
        self.assertTrue(a.llm_used)
        self.assertEqual(a.tier, "capable_llm")
        self.assertEqual(cheap.calls, 1)
        self.assertEqual(capable.calls, 1)

    def test_medium_never_touches_capable(self):
        # A reasoning cue over a solid structured fact -> MEDIUM (cheap only).
        cheap, capable = cheap_answering(), capable_answering()
        sa = self.fresh(cheap=cheap, capable=capable)
        a = sa.ask("Compare the ITC and annual return treatment of Pinnacle Advisory")
        self.assertEqual(a.difficulty, classifier.MEDIUM)
        self.assertEqual(capable.calls, 0)

    def test_classifier_labels(self):
        sa = self.fresh()
        self.assertEqual(sa.ask(EASY_Q).difficulty, classifier.EASY)
        self.assertEqual(sa.ask(OOS_Q).difficulty, classifier.OUT_OF_SCOPE)
        self.assertEqual(sa.ask(HARD_Q).difficulty, classifier.HARD)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #

class TestCaching(unittest.TestCase):

    def test_exact_repeat_hits_response_cache(self):
        cheap = cheap_answering()
        sa = SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=cheap)
        first = sa.ask(HARD_Q)
        self.assertEqual(first.tier, "cheap_llm")
        second = sa.ask(HARD_Q)                    # exact repeat
        self.assertEqual(second.tier, "response_cache")
        self.assertEqual(second.cost_usd, 0.0)
        self.assertEqual(cheap.calls, 1)           # not called again
        self.assertEqual(second.text, first.text)

    def test_paraphrase_hits_semantic_cache(self):
        cheap = cheap_answering()
        sa = SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=cheap)
        sa.ask("How much was the RCM liability?")
        # same tokens, reordered / re-punctuated -> semantic hit, no new work
        para = sa.ask("the RCM liability, how much was it")
        self.assertEqual(para.tier, "semantic_cache")
        self.assertEqual(para.cost_usd, 0.0)

    def test_semantic_cache_topic_guard(self):
        # 'ITC due date' must NOT be reused to answer 'TDS due date'.
        sa = SmartAgent.load(KNOWLEDGE_DIR)
        itc = sa.ask("what is the ITC filing due date")
        tds = sa.ask("what is the TDS filing due date")
        self.assertIn("20 Jun 2026", itc.text)
        self.assertIn("31 Jul 2026", tds.text)
        self.assertNotEqual(itc.text, tds.text)


# --------------------------------------------------------------------------- #
# Cost reduction
# --------------------------------------------------------------------------- #

class TestCost(unittest.TestCase):

    def test_batch_llm_rate_is_low(self):
        cheap = cheap_answering()
        sa = SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=cheap)
        batch = [EASY_Q, "What is the TAN?", "List the vendors",
                 "How much was the RCM liability?", OOS_Q,
                 "When is the TDS statement due?", "What is the verdict on Pinnacle Advisory?",
                 HARD_Q]
        for q in batch:
            sa.ask(q)
        s = sa.stats()
        # only the single HARD question should have hit a model
        self.assertLessEqual(s["llm_call_rate"], 0.2)
        self.assertGreaterEqual(s["zero_cost_fraction"], 0.8)

    def test_prompt_cache_discount_applied(self):
        cheap = cheap_answering()
        sa = SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=cheap)
        a = sa.ask(HARD_Q)
        self.assertTrue(a.prompt_cache_hit)        # stable system prefix cached
        self.assertGreater(a.cost_usd, 0.0)
        self.assertLess(a.cost_usd, 0.01)          # tiny - minimal context

    def test_context_is_minimised(self):
        from rakshak_agent import context as ctx
        sa = SmartAgent.load(KNOWLEDGE_DIR)
        text, sources = ctx.select_context(HARD_Q, sa.kb, sa.index, sa.embed_index,
                                           token_budget=380)
        from rakshak_agent.llm import estimate_tokens
        self.assertLessEqual(estimate_tokens(text), 480)   # budget + one verdict line
        self.assertTrue(sources)


# --------------------------------------------------------------------------- #
# Fallback behaviour
# --------------------------------------------------------------------------- #

class TestFallback(unittest.TestCase):

    def test_llm_error_falls_back_to_offline(self):
        sa = SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=ExplodingClient())
        a = sa.ask(HARD_Q)                # cheap raises -> cascade catches
        self.assertFalse(a.llm_used)      # graceful: offline answer returned
        self.assertTrue(a.text.strip())

    def test_all_abstain_falls_back_to_offline(self):
        sa = SmartAgent.load(KNOWLEDGE_DIR,
                             cheap_llm=cheap_abstaining("c"),
                             capable_llm=MockLLMClient("cap", "capable",
                                                       handler=lambda s, u: ABSTAIN))
        a = sa.ask(HARD_Q)
        self.assertFalse(a.llm_used)
        self.assertTrue(a.text.strip())

    def test_medium_without_cheap_model_does_not_call_capable(self):
        capable = capable_answering()
        sa = SmartAgent.load(KNOWLEDGE_DIR, capable_llm=capable)  # no cheap
        sa.ask("Compare the ITC and annual return treatment of Pinnacle Advisory")
        self.assertEqual(capable.calls, 0)       # MEDIUM must not use capable


# --------------------------------------------------------------------------- #
# Answer quality
# --------------------------------------------------------------------------- #

class TestQuality(unittest.TestCase):

    def test_offline_answer_matches_deterministic(self):
        from rakshak_agent import Agent
        det = Agent.load(KNOWLEDGE_DIR).ask(EASY_Q)
        smart = SmartAgent.load(KNOWLEDGE_DIR).ask(EASY_Q)
        self.assertEqual(smart.text, det.text)

    def test_llm_answer_is_grounded_and_cited(self):
        captured = {}

        def handler(system, user):
            captured["system"] = system
            captured["user"] = user
            return "Grounded answer. [ITC-2627-MAY-0049]"

        cheap = MockLLMClient("cheap", "cheap", handler=handler,
                              price_in_per_m=0.27, price_out_per_m=1.10)
        sa = SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=cheap)
        a = sa.ask(HARD_Q)
        self.assertTrue(a.llm_used)
        self.assertTrue(a.sources)                       # provenance attached
        self.assertIn("CONTEXT", captured["user"])       # grounded prompt
        self.assertIn("INSUFFICIENT_CONTEXT", captured["system"])  # abstain protocol
        # context was actually built from the reports
        self.assertTrue(any(rid in captured["user"]
                            for rid in ("ITC", "ANN", "NTC", "TDS", "WIRE")))

    def test_determinism_of_smart_path(self):
        # Two fresh agents + same deterministic mock -> identical answer text.
        a1 = SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=cheap_answering()).ask(HARD_Q)
        a2 = SmartAgent.load(KNOWLEDGE_DIR, cheap_llm=cheap_answering()).ask(HARD_Q)
        self.assertEqual(a1.text, a2.text)
        self.assertEqual(a1.tier, a2.tier)


if __name__ == "__main__":
    unittest.main(verbosity=2)
