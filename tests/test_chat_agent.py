#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the conversational tool-using agent (ChatAgent).

Uses a scripted MockChatClient so the tool loop is exercised deterministically
without any network. Verifies: tools return correct grounded data, the loop
dispatches tools and composes a final answer, full conversation memory is
passed to the model, sources are collected, cost is accounted, and the agent
falls back to the deterministic engine when no model is configured.
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

from rakshak_agent import Agent, ChatAgent, MockChatClient, ChatTurn
from rakshak_agent.clock import Clock
from rakshak_agent.tools import ToolKit, tool_specs

KB = os.path.join(_ROOT, "knowledge")
TODAY = datetime.date(2026, 7, 7)


def det():
    return Agent.load(KB, clock=Clock(TODAY))


class TestToolKit(unittest.TestCase):
    def setUp(self):
        self.tk = ToolKit(det())

    def test_specs_are_wellformed(self):
        names = {s["function"]["name"] for s in tool_specs()}
        for n in ("get_deadlines", "get_vendor", "find_verdict", "compute_interest",
                  "get_statutory_windows", "get_notice_position", "search_reports"):
            self.assertIn(n, names)

    def test_get_vendor_grounded(self):
        r = self.tk.dispatch("get_vendor", {"name": "Pinnacle Advisory"})
        self.assertEqual(r["vendor"], "Pinnacle Advisory LLP")
        classes = {v["verdict"] for v in r["verdicts"]}
        self.assertIn("REVERSE", classes)

    def test_compute_interest_matches_report(self):
        r = self.tk.dispatch("compute_interest", {"base": 772000, "annual_rate_pct": 18, "days": 3})
        self.assertEqual(r["interest"], 1142)     # ties to NTC Para 4

    def test_find_verdict_grounded(self):
        r = self.tk.dispatch("find_verdict", {"query": "blocked food and beverage credit"})
        self.assertTrue(any("17(5)" in " ".join(v["citations"]) for v in r["verdicts"]))

    def test_deadlines_realtime(self):
        r = self.tk.dispatch("get_deadlines", {})
        self.assertIn("22 Jul 2026", r["summary"])

    def test_unknown_tool_is_safe(self):
        self.assertIn("error", self.tk.dispatch("nope", {}))

    def test_bad_args_dont_crash(self):
        self.assertIn("error", self.tk.dispatch("compute_interest", {}))


class TestChatLoop(unittest.TestCase):
    def agent(self, script):
        return ChatAgent(det(), chat_client=MockChatClient(script=script), clock=Clock(TODAY))

    def test_tool_call_then_answer(self):
        ca = self.agent([
            ChatTurn(tool_calls=[{"id": "c1", "name": "get_vendor",
                                  "arguments": {"name": "Pinnacle Advisory"}}]),
            ChatTurn(content="Reverse ₹36,000 under R.37. [ITC-2627-MAY-0049 p2]"),
        ])
        r = ca.chat([{"role": "user", "content": "how do I handle Pinnacle?"}])
        self.assertIn("Reverse", r.text)
        self.assertIn("get_vendor", r.tools_used)
        self.assertTrue(r.sources)                 # provenance from the tool result
        self.assertGreater(r.cost_usd, 0.0)
        self.assertEqual(r.tool_iterations, 1)

    def test_direct_answer_no_tool(self):
        ca = self.agent([ChatTurn(content="Hello, ask me about the reports.")])
        r = ca.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(r.tools_used, [])
        self.assertEqual(r.text, "Hello, ask me about the reports.")

    def test_full_conversation_memory_passed(self):
        cc = MockChatClient(script=[ChatTurn(content="ok")])
        ChatAgent(det(), chat_client=cc).chat([
            {"role": "user", "content": "what about Orbit?"},
            {"role": "assistant", "content": "Orbit has an inoperative PAN."},
            {"role": "user", "content": "and the fix?"},
        ])
        roles = [m["role"] for m in cc.seen_messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertIn("inoperative PAN", cc.seen_messages[2]["content"])

    def test_system_prompt_injects_date_and_scope(self):
        cc = MockChatClient(script=[ChatTurn(content="ok")])
        ChatAgent(det(), chat_client=cc, clock=Clock(TODAY)).chat(
            [{"role": "user", "content": "hi"}])
        sysmsg = cc.seen_messages[0]["content"]
        self.assertIn("07 Jul 2026", sysmsg)
        self.assertIn("ANN-2425-FY-0012", sysmsg)
        self.assertIn("do not take decisions", sysmsg.replace("does ", ""))

    def test_multi_tool_iteration(self):
        ca = self.agent([
            ChatTurn(tool_calls=[{"id": "a", "name": "get_deadlines", "arguments": {}}]),
            ChatTurn(tool_calls=[{"id": "b", "name": "get_notice_position", "arguments": {}}]),
            ChatTurn(content="Reply to ASMT-11 by 22 Jul 2026. [NTC-2425-JUL-0003 p1]"),
        ])
        r = ca.chat([{"role": "user", "content": "what's most urgent?"}])
        self.assertEqual(r.tools_used, ["get_deadlines", "get_notice_position"])
        self.assertEqual(r.tool_iterations, 2)


class TestCostOptimisations(unittest.TestCase):
    def test_trivial_lookup_skips_the_model(self):
        cc = MockChatClient(script=[ChatTurn(content="should not be called")])
        r = ChatAgent(det(), chat_client=cc, clock=Clock(TODAY)).chat(
            [{"role": "user", "content": "what is the GSTIN?"}])
        self.assertEqual(cc.calls, 0)              # model never invoked
        self.assertEqual(r.cost_usd, 0.0)
        self.assertIn("27XXXXX1234X1Z5", r.text)

    def test_substantive_question_still_uses_model(self):
        cc = MockChatClient(script=[ChatTurn(content="synthesis")])
        ChatAgent(det(), chat_client=cc).chat(
            [{"role": "user", "content": "how should I handle Pinnacle?"}])
        self.assertEqual(cc.calls, 1)

    def test_substantive_with_trivial_keyword_not_fastpathed(self):
        # regression: a single word like 'deadline' must NOT short-circuit a
        # substantive, entity-named, multi-part question to the deterministic path
        cc = MockChatClient(script=[ChatTurn(content="synthesis")])
        ChatAgent(det(), chat_client=cc).chat([{"role": "user", "content":
            "How should I handle Pinnacle Advisory? Give the reversal amount and deadline."}])
        self.assertEqual(cc.calls, 1)      # reached the model, not fast-pathed

    def test_vendor_question_not_fastpathed(self):
        cc = MockChatClient(script=[ChatTurn(content="synthesis")])
        ChatAgent(det(), chat_client=cc).chat(
            [{"role": "user", "content": "what is the verdict on Orbit Packaging?"}])
        self.assertEqual(cc.calls, 1)

    def test_history_is_windowed(self):
        cc = MockChatClient(script=[ChatTurn(content="ok")])
        ca = ChatAgent(det(), chat_client=cc, max_history=4, fast_path=False)
        long_convo = [{"role": "user", "content": "q%d" % i} if i % 2 == 0
                      else {"role": "assistant", "content": "a%d" % i} for i in range(20)]
        ca.chat(long_convo)
        # system + at most `max_history` messages, and it starts user-first
        self.assertLessEqual(len(cc.seen_messages), 1 + 4)
        self.assertEqual(cc.seen_messages[1]["role"], "user")


class TestFallback(unittest.TestCase):
    def test_offline_no_client_uses_deterministic(self):
        r = ChatAgent(det(), chat_client=None).chat(
            [{"role": "user", "content": "What is the GSTIN?"}])
        self.assertIn("27XXXXX1234X1Z5", r.text)
        self.assertTrue(r.fell_back)
        self.assertEqual(r.cost_usd, 0.0)

    def test_model_error_falls_back(self):
        class Boom(MockChatClient):
            def _chat(self, m, t):
                raise RuntimeError("network")
        # a substantive question (not a fast-path lookup) so it reaches the model
        r = ChatAgent(det(), chat_client=Boom()).chat(
            [{"role": "user", "content": "how should I handle Pinnacle Advisory?"}])
        self.assertTrue(r.fell_back)
        self.assertIn("REVERSE", r.text)

    def test_empty_conversation(self):
        r = ChatAgent(det(), chat_client=MockChatClient(script=[])).chat([])
        self.assertTrue(r.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
