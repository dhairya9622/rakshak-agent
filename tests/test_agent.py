#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outcome-accountability test suite for the Rakshak offline agent.

Uses only the Python standard library (unittest) so it runs with:

    python -m unittest discover -s tests            (from the project root)
    python tests/test_agent.py                      (direct)

Coverage:
  1. KnowledgeBaseIntegrity   - the preprocessed KB is complete & correct
  2. DirectAccuracy           - facts stated verbatim in the reports
  3. InferredAccuracy         - cross-report / entity reasoning
  4. QuestionStyles           - many phrasings, casing, abbreviations
  5. Determinism              - same input -> identical output (agent + preprocess)
  6. OutOfScope               - external/ambiguous Qs declined, no hallucination
  7. EdgeCases                - empty, punctuation, unicode, very long, single-word
  8. NoHallucination          - every in-scope answer is grounded & cited
"""

import io
import os
import sys
import unittest

# Make the project importable regardless of where the test is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# UTF-8 stdout so assertion diffs with ₹/§ never crash the console.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from rakshak_agent import Agent
from rakshak_agent import normalize

KNOWLEDGE_DIR = os.path.join(_ROOT, "knowledge")
INPUT_DIR = os.environ.get("RAKSHAK_INPUT", r"C:/Users/DHAIRYA/Downloads/files (2)")


def _agent():
    return Agent.load(KNOWLEDGE_DIR)


# A single shared agent for the read-only tests (loading is deterministic).
AGENT = _agent()


def ask(q):
    return AGENT.ask(q)


def text(q):
    return AGENT.ask(q).text


# --------------------------------------------------------------------------- #
# 1 - Knowledge base integrity
# --------------------------------------------------------------------------- #

class TestKnowledgeBaseIntegrity(unittest.TestCase):

    def test_five_reports_loaded(self):
        self.assertEqual(len(AGENT.kb.reports), 5)
        ids = {r["report_id"] for r in AGENT.kb.reports}
        for expected in ("ANN-2425-FY-0012", "ITC-2627-MAY-0049",
                         "NTC-2425-JUL-0003", "TDS-2627-Q1-0031", "WIRE-MERIDIAN-0001"):
            self.assertIn(expected, ids)

    def test_ledger_counts_match_glance(self):
        # Every module's decision ledger must have exactly the promised rows.
        counts = {}
        for v in AGENT.kb.verdicts:
            counts[v["module"]] = counts.get(v["module"], 0) + 1
        self.assertEqual(counts.get("ANN"), 6)
        self.assertEqual(counts.get("ITC"), 7)
        self.assertEqual(counts.get("NTC"), 4)
        self.assertEqual(counts.get("TDS"), 8)

    def test_every_verdict_is_classified(self):
        # No ledger row may be left without a canonical class (no hallucination
        # downstream depends on this being complete).
        for v in AGENT.kb.verdicts:
            if v["module"] == "WIRE":
                continue
            self.assertIsNotNone(v["verdict_class"], "unclassified: %s" % v["verdict_id"])
            self.assertIsNotNone(v["verdict_alias"], "no alias: %s" % v["verdict_id"])

    def test_amounts_parse_and_are_sane(self):
        for f in AGENT.kb.facts:
            if f["kind"] == "amount":
                self.assertIsNotNone(f.get("value_number"))
                # sample universe: nothing exceeds ~20 Cr; guard against unit bugs
                self.assertLess(f["value_number"], 50_00_00_000,
                                "suspicious amount %s :: %s" % (f["value"], f["context"][:60]))

    def test_ten_entities_indexed(self):
        names = {e["name"] for e in AGENT.kb.entities}
        for who in ("Pinnacle Advisory LLP", "Vasudha Steel Traders", "Annapurna Caterers"):
            self.assertIn(who, names)
        self.assertEqual(len(AGENT.kb.entities), 10)

    def test_manifest_sanity_flags_all_ok(self):
        for r in AGENT.kb.manifest.get("reports", []):
            self.assertTrue(r["verdicts_ok"], "ledger sanity failed for %s" % r["report_id"])


# --------------------------------------------------------------------------- #
# 2 - Direct accuracy (facts stated verbatim)
# --------------------------------------------------------------------------- #

class TestDirectAccuracy(unittest.TestCase):

    def test_gstin(self):
        self.assertIn("27XXXXX1234X1Z5", text("What is the GSTIN?"))

    def test_tan(self):
        self.assertIn("MUMXXXX21F", text("What is the TAN?"))

    def test_pan(self):
        self.assertIn("AAXXX1234X", text("What is the PAN?"))

    def test_taxpayer_entity(self):
        self.assertIn("Meridian Components Pvt Ltd", text("Who is the taxpayer?"))

    def test_itc_filing_due(self):
        self.assertIn("20 Jun 2026", text("When is the ITC return due?"))

    def test_annual_filing_due(self):
        self.assertIn("31 Dec 2025", text("When is the annual return due?"))

    def test_tds_filing_due(self):
        self.assertIn("31 Jul 2026", text("When is the TDS statement due?"))

    def test_notice_reply_due(self):
        self.assertIn("22 Jul 2026", text("When is the notice reply due?"))

    def test_annual_run_hash(self):
        self.assertIn("b66f", text("What is the run hash of the annual return?"))

    def test_rcm_liability_amount(self):
        self.assertIn("36,000", text("How much was the RCM liability?"))

    def test_rcm_interest_amount(self):
        self.assertIn("6,320", text("interest on reverse charge"))

    def test_para4_interest(self):
        self.assertIn("1,142", text("What is the interest admitted in the notice reply?"))

    def test_total_tds(self):
        # "Total TDS ₹3,15,800" is stated on the TDS cover.
        self.assertIn("3,15,800", text("What is the total TDS for the quarter?"))

    def test_aggregate_turnover(self):
        self.assertIn("18", text("What is the aggregate turnover in the annual return?"))

    def test_annapurna_blocked_verdict(self):
        t = text("What is the ITC verdict on Annapurna Caterers?")
        self.assertTrue("BLOCK" in t or "DENY" in t, t)

    def test_report_count(self):
        self.assertIn("5", text("How many reports are there?"))

    def test_vendor_count(self):
        self.assertIn("10", text("How many vendors are tracked?"))

    def test_notice_para_count(self):
        self.assertIn("4", text("How many decisions are in the notice reply?"))


# --------------------------------------------------------------------------- #
# 3 - Inferred accuracy (cross-report / entity reasoning)
# --------------------------------------------------------------------------- #

class TestInferredAccuracy(unittest.TestCase):

    def test_pinnacle_spans_two_verdicts(self):
        # Pinnacle is RECONCILED in the annual and REVERSE in the May ITC run.
        t = text("What is the verdict on Pinnacle Advisory?")
        self.assertIn("REVERSE", t)
        self.assertTrue("RECONCILED" in t or "PASS" in t, t)

    def test_pinnacle_cross_report_sources(self):
        ans = ask("Tell me about the Pinnacle thread")
        rids = {s["report_id"] for s in ans.sources}
        self.assertGreaterEqual(len(rids), 2, "expected multi-report provenance")

    def test_vasudha_spans_three_modules(self):
        ans = ask("Tell me about Vasudha Steel Traders")
        mods = {s["report_id"].split("-")[0] for s in ans.sources}
        # Vasudha appears in ITC, ANN and TDS ledgers.
        self.assertTrue({"ITC", "ANN", "TDS"}.issubset(mods), mods)

    def test_why_blocked_gives_reason(self):
        t = text("Why was Annapurna Caterers blocked?")
        # reason prose from the ledger, not just the class label
        self.assertIn("17(5)", t.replace("§", ""))

    def test_defer_definition(self):
        t = text("What does DEFER mean?")
        self.assertIn("HOLD", t)

    def test_block_definition(self):
        t = text("What does BLOCK mean?")
        self.assertIn("DENY", t)

    def test_repeat_offender_inference(self):
        # The reversal + repeat-offender story lives on Pinnacle.
        t = text("Which vendor is a repeat offender?").lower()
        self.assertIn("pinnacle", t)


# --------------------------------------------------------------------------- #
# 4 - Question-style variety (paraphrase / casing / abbreviation robustness)
# --------------------------------------------------------------------------- #

class TestQuestionStyles(unittest.TestCase):

    def test_gstin_phrasings(self):
        for q in ["What is the GSTIN?", "gstin?", "tell me the gstin number",
                  "GSTIN of Meridian", "what's the gst identification number"]:
            self.assertIn("27XXXXX1234X1Z5", text(q), "failed for %r" % q)

    def test_rcm_amount_phrasings(self):
        for q in ["How much was the RCM liability?",
                  "what is the reverse charge amount",
                  "reverse charge mechanism liability value",
                  "how much RCM tax was missed"]:
            self.assertIn("36,000", text(q), "failed for %r" % q)

    def test_casing_insensitive(self):
        self.assertIn("27XXXXX1234X1Z5", text("WHAT IS THE GSTIN?"))
        self.assertIn("27XXXXX1234X1Z5", text("wHaT iS tHe GsTiN?"))

    def test_entity_casing(self):
        t = text("pInNaClE aDvIsOrY")
        self.assertIn("Pinnacle Advisory LLP", t)

    def test_abbreviation_expansion(self):
        # 'ITC' should route to the input-tax-credit report
        ans = ask("what is the ITC filing deadline")
        self.assertIn("20 Jun 2026", ans.text)


# --------------------------------------------------------------------------- #
# 5 - Determinism
# --------------------------------------------------------------------------- #

class TestDeterminism(unittest.TestCase):

    QUESTIONS = [
        "How much was the RCM liability?",
        "What is the verdict on Pinnacle Advisory?",
        "List the vendors",
        "When is the TDS statement due?",
        "Why was Annapurna Caterers blocked?",
        "What was the weather in Paris?",
    ]

    def test_repeated_ask_identical(self):
        for q in self.QUESTIONS:
            first = AGENT.ask(q).to_dict()
            for _ in range(5):
                self.assertEqual(AGENT.ask(q).to_dict(), first, "non-deterministic: %r" % q)

    def test_independent_agents_identical(self):
        a2 = _agent()
        for q in self.QUESTIONS:
            self.assertEqual(AGENT.ask(q).to_dict(), a2.ask(q).to_dict(),
                             "agent instances diverge: %r" % q)

    def test_whitespace_normalisation_stable(self):
        base = AGENT.ask("How much was the RCM liability?").to_dict()
        spaced = AGENT.ask("   How much was the RCM liability?   ").to_dict()
        self.assertEqual(base, spaced)

    def test_preprocessing_is_byte_identical(self):
        # End-to-end determinism of the pipeline the user actually runs.
        if not os.path.isdir(INPUT_DIR):
            self.skipTest("input PDFs not available at %s" % INPUT_DIR)
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            self.skipTest("pdfplumber not installed")
        import tempfile
        import preprocess

        outs = []
        for _ in range(2):
            d = tempfile.mkdtemp(prefix="rakshak_kb_")
            preprocess.build(INPUT_DIR, d)
            outs.append(d)
        for name in ("reports", "chunks", "verdicts", "facts", "entities"):
            with open(os.path.join(outs[0], name + ".json"), "rb") as fa, \
                 open(os.path.join(outs[1], name + ".json"), "rb") as fb:
                self.assertEqual(fa.read(), fb.read(), "%s.json not deterministic" % name)


# --------------------------------------------------------------------------- #
# 6 - Out-of-scope / ambiguous handling (no hallucination)
# --------------------------------------------------------------------------- #

class TestOutOfScope(unittest.TestCase):

    EXTERNAL = [
        "What was the weather in Paris yesterday?",
        "Who won the 2022 world cup?",
        "What is the capital of France?",
        "What is 2 + 2?",
        "Tell me a joke",
        "asdfghjkl qwerty zxcvbn",
        "What is the meaning of life in 2026?",
    ]

    def test_external_questions_declined(self):
        for q in self.EXTERNAL:
            ans = ask(q)
            self.assertFalse(ans.in_scope, "should be out of scope: %r -> %s" % (q, ans.text[:80]))
            self.assertEqual(ans.intent, "out_of_scope")

    def test_declined_answers_carry_no_fabricated_numbers(self):
        # A refusal must not invent rupee amounts / specifics.
        for q in self.EXTERNAL:
            ans = ask(q)
            self.assertNotIn("₹", ans.text)
            self.assertIn("Out of scope", ans.text)

    def test_declined_answers_have_no_sources(self):
        for q in self.EXTERNAL:
            self.assertEqual(ask(q).sources, [])

    def test_scope_message_lists_known_reports(self):
        ans = ask("Who is the prime minister of India?")
        self.assertFalse(ans.in_scope)
        self.assertTrue(any(m in ans.text for m in ("ITC", "ANN", "TDS", "NTC")))


# --------------------------------------------------------------------------- #
# 7 - Edge cases
# --------------------------------------------------------------------------- #

class TestEdgeCases(unittest.TestCase):

    def test_empty_query(self):
        ans = ask("")
        self.assertEqual(ans.intent, "empty")
        self.assertTrue(ans.text)

    def test_whitespace_only(self):
        self.assertEqual(ask("     ").intent, "empty")

    def test_punctuation_only(self):
        ans = ask("???!!!")
        self.assertFalse(ans.in_scope)

    def test_single_entity_token(self):
        self.assertIn("Pinnacle Advisory LLP", text("Pinnacle"))

    def test_unicode_rupee_in_query(self):
        # question containing ₹ should not crash and should still find the item
        self.assertIn("36,000", text("is ₹36,000 the RCM liability?"))

    def test_very_long_query_is_handled(self):
        q = ("please tell me in great detail " * 40) + "what is the GSTIN"
        ans = ask(q)
        self.assertIn("27XXXXX1234X1Z5", ans.text)

    def test_help_capabilities(self):
        ans = ask("what can you do?")
        self.assertEqual(ans.intent, "capabilities")
        self.assertIn("reports", ans.text.lower())

    def test_greeting(self):
        self.assertEqual(ask("hello").intent, "capabilities")

    def test_answer_is_clean_text(self):
        for q in ["What is the GSTIN?", "List the vendors", "Why was Pinnacle reversed?"]:
            t = text(q)
            self.assertIsInstance(t, str)
            self.assertTrue(t.strip())
            self.assertNotIn("None", t)


# --------------------------------------------------------------------------- #
# 8 - No-hallucination invariants
# --------------------------------------------------------------------------- #

class TestNoHallucination(unittest.TestCase):

    IN_SCOPE = [
        "What is the GSTIN?",
        "How much was the RCM liability?",
        "What is the verdict on Pinnacle Advisory?",
        "When is the TDS statement due?",
        "List the vendors",
        "How many decisions are in the notice reply?",
        "Why was Annapurna Caterers blocked?",
    ]

    def test_in_scope_answers_are_cited(self):
        valid_ids = {r["report_id"] for r in AGENT.kb.reports}
        for q in self.IN_SCOPE:
            ans = ask(q)
            self.assertTrue(ans.in_scope, q)
            self.assertTrue(ans.sources, "no provenance for %r" % q)
            for s in ans.sources:
                self.assertIn(s["report_id"], valid_ids,
                              "cited unknown report %s" % s["report_id"])

    def test_amounts_in_answers_exist_in_knowledge(self):
        # Any rupee amount the agent prints must exist as a real fact/verdict,
        # i.e. it is retrieved, never computed or invented.
        import re
        known = set()
        for f in AGENT.kb.facts:
            if f["kind"] == "amount":
                known.add(f["value"].replace("₹", "").strip())
        for v in AGENT.kb.verdicts:
            for a in v.get("amounts", []):
                known.add(a["raw"].replace("₹", "").strip())
        amount_re = re.compile(r"\d{1,3}(?:,\d{2,3})+")
        for q in self.IN_SCOPE:
            for m in amount_re.findall(ask(q).text):
                self.assertIn(m, known, "answer to %r shows unknown amount %s" % (q, m))

    def test_confidence_monotonic_scope(self):
        # In-scope answers carry positive confidence; refusals carry zero.
        self.assertGreater(ask("What is the GSTIN?").confidence, 0.0)
        self.assertEqual(ask("What is the capital of France?").confidence, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
