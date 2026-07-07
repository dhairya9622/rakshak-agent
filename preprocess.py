#!/usr/bin/env python3
"""
Rakshak offline knowledge preprocessor
======================================

Runs LOCALLY on a folder of Rakshak Systems report PDFs and turns them into
clean, queryable knowledge files that the offline agent loads.

    python preprocess.py --input "C:/Users/DHAIRYA/Downloads/files (2)" --out knowledge

It is deterministic: same PDFs in -> byte-identical JSON out (no timestamps,
no randomness, stable ordering). No network, no LLM, no external services.

It is grammar-aware: it understands the fixed Rakshak "Report Grammar v1.0"
page-role map (P1 cover+verdict, P2 decision ledger, P3 shield+outputs,
P4 drill-down, P-last provenance) and the canonical verdict taxonomy, so it
can pull structured facts and verdicts, not just raw text.

Output files (all JSON, UTF-8):
    manifest.json   - build summary + per-report sanity checks
    reports.json    - one record per report (identity, dates, status, glance...)
    chunks.json     - retrieval units (section-level text blocks)
    verdicts.json   - decision-ledger rows (item, amount, verdict class, cites)
    facts.json      - atomic facts (amounts, hashes, gates, cross-refs, keyed)
    entities.json   - parties/vendors indexed across every report

The agent never re-reads the PDFs; these files ARE the knowledge base.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

try:
    import pdfplumber
except ImportError:  # pragma: no cover - environment guard
    sys.stderr.write(
        "ERROR: pdfplumber is required.\n"
        "Install it once (offline-cached wheel is fine):\n"
        "    python -m pip install pdfplumber\n"
    )
    raise


# --------------------------------------------------------------------------- #
# Canonical verdict taxonomy (Report Grammar section 9).
# Modules use aliases; the pill class + colour + semantics are fixed.
# --------------------------------------------------------------------------- #

# alias (upper, punctuation-normalised) -> (canonical class, colour, meaning)
VERDICT_TAXONOMY = OrderedDict([
    ("CLAIM",       ("PASS",     "green",         "credit/claim allowed")),
    ("RECONCILED",  ("PASS",     "green",         "reconciled clean, no liability")),
    ("CLEAR",       ("PASS",     "green",         "deduction correct, no action")),
    ("EXPLAIN",     ("PASS",     "green",         "explained with evidence, no liability")),
    ("FIX",         ("REPAIRED", "green-outline", "auto-repaired, then allowed")),
    ("DEFER",       ("HOLD",     "amber",         "held back deliberately, protected")),
    ("DISCLOSED",   ("HOLD",     "amber",         "disclosure only, no ledger debit")),
    ("NIL-DOC",     ("HOLD",     "amber",         "nil but documented / watch")),
    ("NIL - DOC",   ("HOLD",     "amber",         "nil but documented / watch")),
    ("NIL DOC",     ("HOLD",     "amber",         "nil but documented / watch")),
    ("WATCH",       ("HOLD",     "amber",         "clean now, flagged for a future window")),
    ("BLOCK",       ("DENY",     "ink",           "blocked credit, cost to P&L")),
    ("LAPSE",       ("DENY",     "ink",           "eligibility lapsed / window closed")),
    ("PAY NOW",     ("PAY",      "red",           "pay before filing")),
    ("PAY",         ("PAY",      "red",           "pay / admitted liability")),
    ("ADMIT + PAY", ("PAY",      "red",           "admission priced to statute, pay")),
    ("ADMIT+PAY",   ("PAY",      "red",           "admission priced to statute, pay")),
    ("PAID",        ("SETTLED",  "blue",          "already paid pre-filing/pre-notice")),
    ("REVERSE",     ("REVERSE",  "red-outline",   "self-reverse now, reclaim on payment")),
    ("RECOVER",     ("REVERSE",  "red-outline",   "recover from counterparty")),
    ("CONTEST",     ("CONTEST",  "navy",          "contest / adjudication posture")),
    ("CA REVIEW",   ("ESCALATE", "amber-outline", "stackable modifier: needs CA sign-off")),
])

# Longest aliases first so "PAY NOW" wins over "PAY", "ADMIT + PAY" over nothing.
VERDICT_ALIASES = sorted(VERDICT_TAXONOMY.keys(), key=len, reverse=True)

MODULE_LABELS = {
    "ITC": "Input Tax Credit - monthly GSTR-3B pre-filing review",
    "ANN": "GSTR-9 & 9C - annual return & reconciliation approval",
    "TDS": "TDS / 26AS / vendor risk - quarterly Form 140 review",
    "NTC": "Notice / DRC / scrutiny - ASMT-10 decode & draft reply",
    "GSTR9": "GSTR-9 & 9C - annual return & reconciliation approval",
    "WIRE": "Integrated workflows - two-year wire across six agents",
    "IRN": "e-Invoice / IRN validation",
    "VND": "Vendor compliance score",
}


# --------------------------------------------------------------------------- #
# Small deterministic text helpers
# --------------------------------------------------------------------------- #

def clean(s):
    """Collapse whitespace, keep the meaningful unicode (₹, ·, §, arrows)."""
    return re.sub(r"[ \t]+", " ", (s or "").replace(" ", " ")).strip()


# Indian-grouped rupee amounts, optionally with Cr / lakh words.
_AMOUNT_RE = re.compile(
    r"₹?\s?"
    r"(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+\.\d+|\d{3,})"   # 1,80,000 / 18.43 / 1142
    r"\s?(Cr|crore|lakhs?|L)?"
    r"(?![A-Za-z0-9])",   # unit must not be the start of a word (e.g. 'credit')
    re.IGNORECASE,
)


def parse_amount_value(number_str, unit):
    """'1,80,000' -> 180000 ; '18.43' + 'Cr' -> 184300000 (paise-free rupees)."""
    n = float(number_str.replace(",", ""))
    if unit:
        u = unit.lower()
        if u in ("cr", "crore"):
            n *= 1e7
        elif u in ("lakh", "lakhs", "l"):
            n *= 1e5
    return int(round(n))


def find_amounts(text):
    """Return [(raw_text, integer_rupees)] for every rupee amount in `text`.

    Money qualifies only when it carries a ₹ sign, Indian comma grouping, or a
    Cr/lakh unit. Bare decimals (section refs like 3.1(d), percentages like
    0.36%) and plain small integers are rejected.
    """
    out = []
    for m in _AMOUNT_RE.finditer(text):
        num, unit = m.group(1), m.group(2)
        raw = m.group(0).strip()
        has_sign = "₹" in raw
        has_group = "," in num
        has_dec = "." in num
        tail = text[m.end():m.end() + 1]
        # reject "3.1(d)" (section), "0.36%" (rate), "1/5" (page) style tails
        if tail in ("%", "(") or (tail == "/" and not has_group):
            continue
        if unit:
            pass  # "₹18.43 Cr" - always money
        elif has_group or has_sign:
            pass  # "1,80,000" / "₹42,320"
        else:
            continue  # bare integer or bare decimal - not money
        out.append((clean(raw), parse_amount_value(num, unit)))
    return out


# Proof/run hashes are printed as 4hex + ellipsis + 4hex, and occasionally
# wrap across a line break (e.g. "proof: a820…\n5ab8"), so allow inner space.
_PROOF_RE = re.compile(r"proof:\s*([0-9a-f]{4})\s*[.…]{1,3}\s*([0-9a-f]{4})", re.IGNORECASE)
_HASH_RE = re.compile(r"(?:sha256:)?([0-9a-f]{4})\s*[.…]{1,3}\s*([0-9a-f]{4})", re.IGNORECASE)
_REPORTID_RE = re.compile(r"\b((?:ITC|ANN|GSTR9|TDS|NTC|IRN|VND|WIRE)[-A-Z0-9]*-\d{3,4})\b")
_GATE_RE = re.compile(r"\b((?:ITC|TDS|NTC|ANN)\.G\d{1,3})\b")
_DIN_RE = re.compile(r"\bDIN[-A-Z0-9]+\b")
_ARN_RE = re.compile(r"\bARN\s+([A-Z0-9]{10,})\b")

# Statutory citations: §.., Rule/R..., Table/T..., Circ..., Notif..., NSxx, forms.
_CITE_RE = re.compile(
    r"(§\s?\d+[0-9A-Za-z()/. ]*?(?=·|proof|Annex|$)"
    r"|R\.\s?\d+[0-9A-Za-z()]*"
    r"|Rule\s+\d+[0-9A-Za-z()]*"
    r"|Circ\.\s?\d+[0-9/]*"
    r"|Notif\.\s?\d+[0-9/\-A-Za-z. ]*?(?=·|$)"
    r"|(?:GSTR-9C?|9C)\s?(?:Part\s?[VIX]+|T\.?\s?\d+[A-Z0-9/]*)"
    r"|T\.\s?\d+[A-Z0-9/]*"
    r"|Table\s+\d+[A-Z0-9/]*"
    r"|NS\d{2}(?:/\d{2})?)",
)


def find_citations(text):
    seen = []
    for m in _CITE_RE.finditer(text):
        c = clean(m.group(1)).rstrip(" ·")
        if c and c not in seen:
            seen.append(c)
    return seen


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #

def extract_pages(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def module_from_id(report_id):
    m = re.match(r"([A-Z0-9]+?)-", report_id or "")
    if m:
        code = m.group(1)
        return "ANN" if code == "GSTR9" else code
    return "UNK"


# --------------------------------------------------------------------------- #
# Report-level parsing (P1 masthead + identity + glance + P-last provenance)
# --------------------------------------------------------------------------- #

def _first(regex, text, group=1, flags=0, default=None):
    m = re.search(regex, text, flags)
    return clean(m.group(group)) if m else default


def parse_report(pages, filename):
    p1 = pages[0]
    plast = pages[-1]
    joined = "\n".join(pages)

    report_id = _first(_REPORTID_RE, p1) or _first(_REPORTID_RE, joined) or "UNKNOWN"
    module = module_from_id(report_id)

    rec = OrderedDict()
    rec["report_id"] = report_id
    rec["module"] = module
    rec["module_label"] = MODULE_LABELS.get(module, module)
    rec["source_file"] = filename
    rec["pages"] = len(pages)

    # Module/title line: the ALL-CAPS product line under the masthead.
    title = None
    for line in p1.splitlines():
        lc = clean(line)
        if lc.isupper() and 8 <= len(lc) <= 80 and "REPORT" not in lc and "SAMPLE" not in lc:
            title = lc
            break
    rec["title"] = title

    rec["entity"] = _first(r"(Meridian Components Pvt Ltd)", joined) \
        or _first(r"^([A-Z][A-Za-z .&]+ (?:Pvt Ltd|Ltd|LLP))", p1, flags=re.M)
    rec["gstin"] = _first(r"GSTIN\s+([0-9A-Z]{15}|[0-9A-Zx]{15})", joined)
    rec["tan"] = _first(r"TAN\s+([A-Z0-9x]{9,})", joined)
    rec["pan"] = _first(r"PAN\s+([A-Z0-9x]{10})", joined)

    rec["generated"] = _first(r"Generated\s+([0-9]{1,2}\s+\w+\s+\d{4}\s*·\s*[0-9:]+\s*IST)", joined)
    rec["status"] = _first(r"STATUS:\s*([A-Z0-9 ·]+?)(?:\s{2,}|RECONCILED\b|ISSUED\b|$)", p1, flags=re.M)
    rec["filing_due"] = _first(r"FILING DUE\s+([0-9]{1,2}\s+\w+\s+\d{4})", joined)
    rec["reply_due"] = _first(r"REPLY DUE\s+([0-9]{1,2}\s+\w+\s+\d{4})", joined)
    rec["period"] = (_first(r"\b(FY\s?20\d\d[-/]?\d\d)\b", joined)
                     or _first(r"\b(Q[1-4]\s*·?\s*(?:FY|TY|Tax Year)\s?20\d\d[-/]?\d*)", joined))

    rec["engine_version"] = _first(r"(?:verdict|response)\s+v(\d+\.\d+\.\d+)", joined)
    rec["engine_version_line"] = _first(r"(SentinelXOS[^\n]*v\d+\.\d+\.\d+)", joined) \
        or _first(r"((?:Annual|ITC|TDS|Notice|TDS·26AS)[^\n]*v\d+\.\d+\.\d+)", joined)
    rec["run_hash"] = _first(r"RUN HASH\s+(sha256:[0-9a-f]{4}[.…]+[0-9a-f]{4})", joined)

    # Glance sentence (P1): "You decide N ... reaches your desk." / notice variant.
    rec["glance_sentence"] = (
        _first(r"(You decide[^\n]*?desk\.)", p1)
        or _first(r"(Officer alleged[^\n]*?DRC-03\.)", joined)
    )
    m = re.search(r"You decide\s+([\d,]+)\s+(?:items?|lines?)\.\s*"
                  r"The engine (?:cleared|reconciled)\s+([\d,]+)", p1)
    if m:
        rec["decisions_to_desk"] = int(m.group(1).replace(",", ""))
        rec["engine_cleared"] = int(m.group(2).replace(",", ""))
    else:
        rec["decisions_to_desk"] = None
        rec["engine_cleared"] = None
    # Notice module has no "You decide N" line; its ledger is the raised paras.
    mp = re.search(r"PARAS RAISED\s+(\d+)\s+discrepanc", joined)
    if mp and rec["decisions_to_desk"] is None:
        rec["decisions_to_desk"] = int(mp.group(1))

    # Epigraph: the italic serif line just above the P1 footer.
    rec["epigraph"] = _epigraph(p1)

    rec["basis"] = _first(r"BASIS\s+([^\n]+)", joined)
    rec["inputs"] = _first(r"INPUTS\s+([^\n]+)", plast)
    rec["consumes"] = _first(r"CONSUMES\s+([^\n]+)", plast)
    rec["publishes"] = _first(r"PUBLISHES\s+([^\n]+)", plast)

    return rec


_KNOWN_EPIGRAPHS = [
    "Every credit carries its birthdate — the annual return compiles itself.",
    "Every deduction carries its section, its challan, and its mirror in someone else's 26AS.",
    "The reply is not drafted — it is retrieved.",
    "The annual return is not prepared — it is folded from twelve monthly ledgers.",
]


def _epigraph(p1):
    for ep in _KNOWN_EPIGRAPHS:
        core = ep.split("—")[0].strip()[:20]
        if core and core in p1:
            # Return the exact line as printed (handles the notice extension).
            for line in p1.splitlines():
                if core in line:
                    return clean(line)
    return None


# --------------------------------------------------------------------------- #
# Verdict-ledger parsing (P2 - always the decision ledger in every module)
# --------------------------------------------------------------------------- #

def _strip_pill_legend(segment):
    """
    Drop the P2 key-strip legend line (e.g. 'CLAIM FIX DEFER BLOCK REVERSE
    CA REVIEW') that lists every verdict alias and would otherwise be mistaken
    for the first row's verdict.
    """
    kept = []
    for line in segment.splitlines():
        norm = re.sub(r"[–—−]", "-", line.upper())
        stripped = norm
        for alias in VERDICT_ALIASES:
            stripped = re.sub(r"(?<![A-Z])" + re.escape(alias) + r"(?![A-Z])", " ", stripped)
        # A legend line is >=2 aliases and nothing else alphabetic left over.
        n_aliases = sum(1 for a in VERDICT_ALIASES
                        if re.search(r"(?<![A-Z])" + re.escape(a) + r"(?![A-Z])", norm))
        residue = re.sub(r"[^A-Z]", "", stripped)
        if n_aliases >= 2 and not residue:
            continue
        kept.append(line)
    return "\n".join(kept)


def detect_verdict(segment):
    """Return (alias, canonical_class, colour, meaning, is_modifier) or Nones.

    The verdict pill sits between the gate-checks and the reason prose, so we
    take the EARLIEST-positioned alias (after removing the legend line and the
    stackable CA-REVIEW modifier).
    """
    body = _strip_pill_legend(segment)
    up = re.sub(r"[–—−]", "-", body.upper())
    up = re.sub(r"[·|]", " ", up)

    best_alias, best_pos = None, None
    for alias in VERDICT_ALIASES:
        if alias == "CA REVIEW":
            continue  # stackable modifier, handled separately
        m = re.search(r"(?<![A-Z])" + re.escape(alias) + r"(?![A-Z])", up)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_alias = m.start(), alias

    modifier = bool(re.search(r"(?<![A-Z])CA REVIEW(?![A-Z])", up))
    if not best_alias:
        return (None, None, None, None, modifier)
    cls, colour, meaning = VERDICT_TAXONOMY[best_alias]
    return (best_alias, cls, colour, meaning, modifier)


def parse_ledger(page2, report):
    """
    Split the P2 ledger stream on proof hashes; each piece is one decision.
    Robust to column interleaving because every ledger row carries exactly one
    'proof: xxxx…xxxx' anchor.
    """
    verdicts = []
    anchors = list(_PROOF_RE.finditer(page2))
    if not anchors:
        return verdicts

    start = 0
    # Skip the header band (before the first row's content) heuristically by
    # letting the first segment run from page start to the first proof anchor.
    for i, m in enumerate(anchors):
        seg = page2[start:m.end()]
        start = m.end()
        proof = "%s…%s" % (m.group(1).lower(), m.group(2).lower())

        alias, cls, colour, meaning, modifier = detect_verdict(seg)
        amts = find_amounts(seg)
        cites = find_citations(seg)

        item = _ledger_item_name(seg, report["module"])
        party = _party_name(seg)

        v = OrderedDict()
        v["verdict_id"] = "%s#v%d" % (report["report_id"], i + 1)
        v["report_id"] = report["report_id"]
        v["module"] = report["module"]
        v["item"] = item
        v["party"] = party
        v["verdict_alias"] = alias
        v["verdict_class"] = cls
        v["verdict_color"] = colour
        v["verdict_meaning"] = meaning
        v["ca_review"] = modifier
        v["amounts"] = [{"raw": r, "value": val} for r, val in amts]
        v["primary_amount"] = amts[0][1] if amts else None
        v["citations"] = cites
        v["proof"] = proof
        v["text"] = clean(seg.replace("\n", " "))
        verdicts.append(v)
    return verdicts


# Party / vendor names seen in the sample universe. Used to anchor ledger rows
# and to build the cross-report entity index. (Derived from the reports; the
# preprocessor also picks up any GSTIN-tagged name it has not seen before.)
KNOWN_PARTIES = [
    "Rashmi Infotech Pvt Ltd", "Orbit Packaging Co", "Speedex Logistics",
    "Annapurna Caterers", "Vasudha Steel Traders", "Nimbus Electricals",
    "Pinnacle Advisory LLP", "Stratus Advisory Services", "Horizon Estates",
    "Zenith Fabricators",
]


def _party_name(seg):
    for p in KNOWN_PARTIES:
        if p in seg:
            return p
    # Fall back to a leading Proper-Noun run (up to a company suffix).
    m = re.match(r"\s*([A-Z][A-Za-z&.]+(?:\s+[A-Z][A-Za-z&.]+){0,3}"
                 r"(?:\s+(?:Pvt Ltd|Ltd|LLP|Co|Services|Estates|Caterers|Logistics|Traders|Fabricators|Electricals)))",
                 seg)
    return clean(m.group(1)) if m else None


_HEADER_LINE_RE = re.compile(
    r"(decisions?\b|ITEM\s+AMOUNT|PARA\s*/\s*ALLEGATION|DEDUCTEE\s*/|SUPPLIER\s*/|"
    r"GATE CHECKS|REASON & STATUTORY|RESPONSE & STATUTORY|TAXABLE)", re.IGNORECASE)


def _ledger_item_name(seg, module):
    """A short human label for the decision (para / party / item heading)."""
    # Notice rows are 'Para N — allegation'.
    para = re.search(r"(Para\s+\d+\s*[—-][^\n·]{3,60})", seg)
    if para:
        # keep the allegation phrase, drop any trailing grouped amount
        return re.sub(r"\s+₹?\d{1,3}(?:,\d{2,3})+.*$", "", clean(para.group(1)))[:70]
    party = _party_name(seg)
    if party:
        return party
    # Item-based rows (ANN): first content line that is not a header/legend,
    # trimmed at the first amount / gate tick / column gap.
    for raw in seg.splitlines():
        line = clean(raw)
        if not line or _HEADER_LINE_RE.search(line):
            continue
        if line.upper() == line and " " in line and "₹" not in line:
            continue  # a stray legend / all-caps band
        label = re.split(r"\s(?=₹?\d{1,3}(?:,\d{2,3}))|[✓✗]| {2,}", line)[0]
        label = clean(label)
        if len(label) >= 3:
            return label[:70]
    return None


# --------------------------------------------------------------------------- #
# Chunking (section-level text blocks for retrieval)
# --------------------------------------------------------------------------- #

# Fixed page roles from the grammar (4-page monthly/quarterly, 5-page annual/notice).
def page_role(page_no, total):
    if page_no == 1:
        return "Cover & verdict at a glance"
    if page_no == 2:
        return "Decision ledger"
    if page_no == 3:
        return "Shield & outputs"
    if page_no == total:
        return "Provenance & sign-off"
    return "Drill-down / annexure"


_HEADING_HINTS = [
    "at a glance", "shield", "exactly what files", "filing payload", "reply payload",
    "memo", "prepared by", "basis of preparation", "annexure", "drill-down",
    "ledger", "clock", "pipeline", "related modules", "response gates",
    "crosswalk", "gate", "vendor risk", "waterfall", "draft reply",
]


def is_heading(line):
    lc = clean(line)
    if not lc:
        return False
    low = lc.lower()
    if any(h in low for h in _HEADING_HINTS):
        return len(lc) <= 70
    words = lc.split()
    if len(words) <= 8 and lc == lc.upper() and any(c.isalpha() for c in lc) and "₹" not in lc:
        return True
    return False


def chunk_pages(pages, report):
    chunks = []
    total = len(pages)
    for pno, text in enumerate(pages, start=1):
        role = page_role(pno, total)
        lines = [l for l in text.splitlines()]
        blocks = []
        cur_head = clean(lines[0]) if lines else role
        cur_body = []
        for line in lines[1:]:
            if is_heading(line) and cur_body:
                blocks.append((cur_head, "\n".join(cur_body)))
                cur_head = clean(line)
                cur_body = []
            else:
                cur_body.append(line)
        if cur_body or cur_head:
            blocks.append((cur_head, "\n".join(cur_body)))

        for bi, (head, body) in enumerate(blocks):
            body_c = clean(body.replace("\n", " "))
            if not body_c and not head:
                continue
            c = OrderedDict()
            c["chunk_id"] = "%s#p%d#b%d" % (report["report_id"], pno, bi)
            c["report_id"] = report["report_id"]
            c["module"] = report["module"]
            c["page"] = pno
            c["page_role"] = role
            c["section"] = head
            c["text"] = clean((head + ". " + body_c) if head else body_c)
            chunks.append(c)
    return chunks


# --------------------------------------------------------------------------- #
# Fact extraction (amounts, hashes, gates, cross-references, keyed identity)
# --------------------------------------------------------------------------- #

def _sentences(text):
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z0-9₹])", text)
    return [clean(p) for p in parts if clean(p)]


def collect_facts(pages, report):
    facts = []
    rid = report["report_id"]
    mod = report["module"]

    def add(kind, subject, value, value_number, page, context, tags):
        f = OrderedDict()
        f["fact_id"] = "%s#f%d" % (rid, len(facts) + 1)
        f["report_id"] = rid
        f["module"] = mod
        f["kind"] = kind
        f["subject"] = subject
        f["value"] = value
        f["value_number"] = value_number
        f["page"] = page
        f["context"] = context
        f["tags"] = tags
        facts.append(f)

    # Keyed identity facts (deterministic, high-precision).
    keyed = [
        ("report id", report.get("report_id")),
        ("module", report.get("module_label")),
        ("entity", report.get("entity")),
        ("GSTIN", report.get("gstin")),
        ("TAN", report.get("tan")),
        ("PAN", report.get("pan")),
        ("status", report.get("status")),
        ("filing due date", report.get("filing_due")),
        ("reply due date", report.get("reply_due")),
        ("period", report.get("period")),
        ("generated", report.get("generated")),
        ("engine version", report.get("engine_version_line")),
        ("run hash", report.get("run_hash")),
        ("basis", report.get("basis")),
        ("glance", report.get("glance_sentence")),
        ("epigraph", report.get("epigraph")),
        ("decisions to desk", report.get("decisions_to_desk")),
        ("engine cleared count", report.get("engine_cleared")),
    ]
    for subj, val in keyed:
        if val in (None, ""):
            continue
        add("identity", subj, str(val),
            val if isinstance(val, int) else None, 1,
            "%s: %s" % (subj, val),
            _tags(subj + " " + str(val)))

    # Amount facts with sentence context, across every page. Insert a sentence
    # break after each proof hash so interleaved P2 ledger rows split cleanly.
    for pno, text in enumerate(pages, start=1):
        flat = clean(text.replace("\n", " "))
        flat = _PROOF_RE.sub(lambda m: "proof:%s…%s. " % (m.group(1), m.group(2)), flat)
        for sent in _sentences(flat):
            amts = find_amounts(sent)
            if not amts:
                continue
            for raw, val in amts:
                add("amount", None, raw, val, pno, sent, _tags(sent))

    # Cross-reference / provenance facts (report-wide, page 1 anchor).
    joined = "\n".join(pages)
    for rep in sorted(set(_REPORTID_RE.findall(joined))):
        if rep != rid:
            add("cross_reference", "related report", rep, None, 1,
                "%s references %s" % (rid, rep), ["related", "reference", rep.lower()])
    for gate in sorted(set(_GATE_RE.findall(joined))):
        ctx = _first(re.escape(gate) + r"\s*[·:]?\s*([^\n]{0,90})", joined) or gate
        add("gate", gate, gate, None, 1, clean(gate + " " + ctx), ["gate", gate.lower()])
    for din in sorted(set(_DIN_RE.findall(joined))):
        add("reference", "DIN", din, None, 1, "Notice DIN %s" % din, ["din", "notice", "reference"])
    for arn in sorted(set(_ARN_RE.findall(joined))):
        add("reference", "DRC-03 ARN", arn, None, 1, "DRC-03 ARN %s" % arn, ["arn", "drc-03", "payment"])

    return facts


_STOP = set("""a an the of to in on for and or is are was were be been with by at as from into per
via not no this that these those it its their his her your you we our up out off over under than then
which who whom whose what when where why how do does did done has have had will would can could should
than only also both each any all some more most much many few both either neither same such very""".split())


def _tags(text):
    """Lowercase content tokens (letters/digits, keep the domain shorthand)."""
    toks = re.findall(r"[a-z0-9][a-z0-9()/.\-]*", text.lower())
    return sorted({t.strip("().-/") for t in toks
                   if len(t) > 1 and t not in _STOP})[:40]


# --------------------------------------------------------------------------- #
# Entity index (parties/vendors across every report)
# --------------------------------------------------------------------------- #

def build_entities(all_verdicts, all_chunks, reports_by_id):
    ents = OrderedDict()
    for p in KNOWN_PARTIES:
        ents[p] = OrderedDict([("name", p), ("aliases", _aliases(p)), ("mentions", [])])

    def touch(name, report_id, page, snippet, verdict_class=None, amount=None):
        if name not in ents:
            ents[name] = OrderedDict([("name", name), ("aliases", _aliases(name)), ("mentions", [])])
        ents[name]["mentions"].append(OrderedDict([
            ("report_id", report_id),
            ("module", reports_by_id.get(report_id, {}).get("module")),
            ("page", page),
            ("verdict_class", verdict_class),
            ("amount", amount),
            ("snippet", snippet[:240]),
        ]))

    for v in all_verdicts:
        if v["party"]:
            touch(v["party"], v["report_id"], 2, v["text"],
                  v["verdict_class"], v.get("primary_amount"))

    # Also scan chunks so narrative-only mentions (e.g. WIRE, shields) are caught.
    for c in all_chunks:
        for name in list(ents.keys()):
            if name in c["text"] or any(a in c["text"] for a in ents[name]["aliases"]):
                # avoid double-counting the ledger row we already have
                if not (c["page"] == 2 and any(
                        m["report_id"] == c["report_id"] and m["page"] == 2
                        for m in ents[name]["mentions"])):
                    touch(name, c["report_id"], c["page"], c["text"])

    # Drop entities never actually seen; stable order.
    return OrderedDict((k, v) for k, v in ents.items() if v["mentions"])


def _aliases(name):
    al = set()
    short = re.sub(r"\s+(Pvt Ltd|Ltd|LLP|Co|Services|Estates|Caterers|Logistics|Traders|Fabricators|Electricals)$",
                   "", name).strip()
    if short and short != name:
        al.add(short)
    first = name.split()[0]
    if len(first) > 3:
        al.add(first)
    return sorted(al)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build(input_dir, out_dir):
    pdfs = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".pdf"))
    if not pdfs:
        raise SystemExit("No PDF files found in %s" % input_dir)

    reports, chunks, verdicts, facts = [], [], [], []
    manifest_reports = []

    for fn in pdfs:
        path = os.path.join(input_dir, fn)
        pages = extract_pages(path)
        report = parse_report(pages, fn)
        reports.append(report)

        vs = parse_ledger(pages[1], report) if len(pages) >= 2 else []
        verdicts.extend(vs)
        cs = chunk_pages(pages, report)
        chunks.extend(cs)
        fs = collect_facts(pages, report)
        facts.extend(fs)

        expected = report.get("decisions_to_desk")
        manifest_reports.append(OrderedDict([
            ("report_id", report["report_id"]),
            ("source_file", fn),
            ("pages", len(pages)),
            ("chunks", len(cs)),
            ("facts", len(fs)),
            ("verdicts_found", len(vs)),
            ("verdicts_expected", expected),
            ("verdicts_ok", (expected is None) or (len(vs) == expected)),
        ]))

    reports_by_id = {r["report_id"]: r for r in reports}
    entities = build_entities(verdicts, chunks, reports_by_id)

    manifest = OrderedDict([
        ("schema_version", "1.0"),
        ("generator", "rakshak preprocess.py"),
        ("input_dir", os.path.abspath(input_dir)),
        ("report_count", len(reports)),
        ("chunk_count", len(chunks)),
        ("verdict_count", len(verdicts)),
        ("fact_count", len(facts)),
        ("entity_count", len(entities)),
        ("reports", manifest_reports),
    ])

    os.makedirs(out_dir, exist_ok=True)
    _dump(os.path.join(out_dir, "manifest.json"), manifest)
    _dump(os.path.join(out_dir, "reports.json"), reports)
    _dump(os.path.join(out_dir, "chunks.json"), chunks)
    _dump(os.path.join(out_dir, "verdicts.json"), verdicts)
    _dump(os.path.join(out_dir, "facts.json"), facts)
    _dump(os.path.join(out_dir, "entities.json"), list(entities.values()))
    return manifest


def _dump(path, obj):
    # sort_keys=False to preserve our deterministic OrderedDict ordering;
    # ensure_ascii=False keeps ₹/§ readable; fixed separators = stable bytes.
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, separators=(",", ": "))
        fh.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Preprocess Rakshak report PDFs into offline knowledge files.")
    ap.add_argument("--input", "-i", default="C:/Users/DHAIRYA/Downloads/files (2)",
                    help="Folder containing the report PDFs.")
    ap.add_argument("--out", "-o", default="knowledge",
                    help="Output folder for the JSON knowledge base.")
    args = ap.parse_args(argv)

    manifest = build(args.input, args.out)

    print("Rakshak knowledge base built ->", os.path.abspath(args.out))
    print("  reports : %d" % manifest["report_count"])
    print("  chunks  : %d" % manifest["chunk_count"])
    print("  verdicts: %d" % manifest["verdict_count"])
    print("  facts   : %d" % manifest["fact_count"])
    print("  entities: %d" % manifest["entity_count"])
    print("  ledger sanity:")
    for r in manifest["reports"]:
        flag = "ok" if r["verdicts_ok"] else "CHECK"
        print("    [%s] %-22s verdicts %s/%s (%s)" % (
            flag, r["report_id"], r["verdicts_found"], r["verdicts_expected"], r["source_file"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
