#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversational-behaviour tests (from a real sample interaction that exposed
gaps): role/boundary questions, advisory follow-ups, anaphora resolution via
optional context, and clarify-don't-refuse for under-specified follow-ups.
Pinned clock for determinism.
"""

import datetime
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

from rakshak_agent import SmartAgent
from rakshak_agent.clock import Clock

KB = os.path.join(_ROOT, "knowledge")
TODAY = datetime.date(2026, 7, 7)


def agent():
    return SmartAgent.load(KB, clock=Clock(TODAY))


class TestRoleBoundary(unittest.TestCase):
    def test_take_decisions_gets_boundary_not_snippet(self):
        a = agent().ask("can you take decisions on my behalf?")
        self.assertEqual(a.intent, "role")
        self.assertIn("don't take decisions", a.text)
        self.assertTrue(a.in_scope)
        self.assertEqual(a.cost_usd, 0.0)

    def test_sign_off_boundary(self):
        self.assertEqual(agent().ask("will you sign it off for me?").intent, "role")

    def test_role_is_zero_cost(self):
        self.assertFalse(agent().ask("can you decide for me?").llm_used)


class TestAdvisoryPhrasings(unittest.TestCase):
    def test_deal_with_entity_gives_item_advice(self):
        a = agent().ask("how do you think we should deal with Pinnacle?")
        self.assertEqual(a.intent, "advice")
        self.assertIn("REVERSE", a.text)
        self.assertIn("30 Nov 2026", a.text)
        self.assertEqual(a.topic, "Pinnacle Advisory LLP")

    def test_next_month_gives_deadlines_not_dump(self):
        a = agent().ask("what do you think I should do next month?")
        self.assertEqual(a.intent, "advice")
        self.assertIn("Due (as of", a.text)      # deadlines, not a waterfall chunk

    def test_this_week_gives_deadlines(self):
        self.assertIn("Due (as of", agent().ask("what should I do this week?").text)


class TestAnaphoraFollowUps(unittest.TestCase):
    def test_followup_without_context_clarifies_not_refuses(self):
        a = agent().ask("can you advise on this?")
        self.assertTrue(a.in_scope)                 # NOT refused
        self.assertNotEqual(a.tier, "refused")
        self.assertIn("which item", a.text.lower())

    def test_followup_with_context_resolves_topic(self):
        a = agent().ask("can you advise on this?",
                        context={"last_entity": "Pinnacle Advisory LLP"})
        self.assertIn("REVERSE", a.text)            # resolved to Pinnacle
        self.assertEqual(a.topic, "Pinnacle Advisory LLP")

    def test_no_topic_bleed_across_context(self):
        sa = agent()
        p = sa.ask("advise on this", context={"last_entity": "Pinnacle Advisory LLP"})
        o = sa.ask("advise on this", context={"last_entity": "Orbit Packaging Co"})
        self.assertIn("Pinnacle", p.text)
        self.assertIn("Orbit", o.text)
        self.assertNotEqual(p.text, o.text)         # cache did not bleed topics

    def test_entity_answer_exposes_topic_for_echo(self):
        a = agent().ask("what is the verdict on Annapurna Caterers?")
        self.assertEqual(a.topic, "Annapurna Caterers")


if __name__ == "__main__":
    unittest.main(verbosity=2)
