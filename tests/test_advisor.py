#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the virtual-CA reasoning layer (deterministic, pinned clock).

Real-time answers change with the date, so we PIN the clock to 2026-07-07 and
assert exact computed day-counts / deadlines. This proves the advisory engine is
deterministic given a date, and that it reasons (counts days, orders, closes
windows) rather than just retrieves.
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

from rakshak_agent import Agent
from rakshak_agent.clock import Clock

KNOWLEDGE_DIR = os.path.join(_ROOT, "knowledge")
TODAY = datetime.date(2026, 7, 7)


def agent():
    return Agent.load(KNOWLEDGE_DIR, clock=Clock(TODAY))


class TestDeadlines(unittest.TestCase):
    def test_priority_lists_upcoming_with_daycount(self):
        t = agent().ask("What should I prioritise?").text
        self.assertIn("ASMT-11 reply — 22 Jul 2026 · 15d", t)   # 22 Jul - 7 Jul
        self.assertIn("Form 140 — 31 Jul 2026 · 24d", t)        # 31 Jul - 7 Jul

    def test_overdue_is_flagged(self):
        t = agent().ask("upcoming deadlines").text
        self.assertIn("OVERDUE", t)

    def test_intent_is_advice(self):
        self.assertEqual(agent().ask("what should I prioritise?").intent, "advice")


class TestWindows(unittest.TestCase):
    def test_r37a_deadline_is_reversal_date(self):
        t = agent().ask("which statutory windows are closing?").text
        self.assertIn("R.37A reversal", t)
        self.assertIn("30 Nov 2026", t)                 # deadline, not 30 Sep condition
        self.assertIn("146d", t)                         # 30 Nov - 7 Jul

    def test_msmed_window(self):
        t = agent().ask("open windows").text
        self.assertIn("MSMED", t)
        self.assertIn("31 Mar 2027", t)


class TestItemAdvice(unittest.TestCase):
    def test_pinnacle_action_and_deadline(self):
        t = agent().ask("what should I do about Pinnacle Advisory?").text
        self.assertIn("REVERSE", t)
        self.assertIn("30 Nov 2026", t)
        self.assertIn("CA review required", t)


class TestNoticePosture(unittest.TestCase):
    def test_posture_counts_paras(self):
        t = agent().ask("what is the notice posture?").text
        self.assertIn("4 paras", t)
        self.assertIn("reply due 22 Jul 2026", t)


class TestDeterminismAndScope(unittest.TestCase):
    def test_same_date_same_advice(self):
        self.assertEqual(agent().ask("what should I prioritise?").text,
                         agent().ask("what should I prioritise?").text)

    def test_advice_is_terse(self):
        # extremely short: priority answer stays compact
        t = agent().ask("what should I prioritise?").text
        self.assertLessEqual(len(t.splitlines()), 6)

    def test_out_of_scope_is_terse_refusal(self):
        a = agent().ask("who won the world cup?")
        self.assertFalse(a.in_scope)
        self.assertIn("Out of scope", a.text)
        self.assertLessEqual(len(a.text), 130)


if __name__ == "__main__":
    unittest.main(verbosity=2)
