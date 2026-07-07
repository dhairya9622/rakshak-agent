"""
Answer composition.

Every answer is EXTRACTIVE: it is built only from values and verbatim sentences
already present in the knowledge base. Nothing is generated or paraphrased, so
the agent cannot hallucinate. This module provides the formatting helpers; the
engine decides which one to call.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from . import normalize


_SENT_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9₹§])")

# Navigation / footer boilerplate that repeats verbatim across every report and
# carries no answer content. Excluded from extractive selection.
_BOILERPLATE_RE = re.compile(
    r"SAME VERDICT LEDGER|RELATED MODULES|IRN validation|Vendor compliance score|"
    r"sample dataset, illustrative figures|SentinelXOS engine —|·\s*pg\s*\d",
    re.IGNORECASE)


def sentences(text: str) -> List[str]:
    out = []
    for part in _SENT_SPLIT_RE.split(text or ""):
        p = part.strip()
        if len(p) >= 8:
            out.append(p)
    return out


def select_sentences(chunks: List[Dict], terms: List[str], limit: int = 3,
                     weights: Dict[str, float] = None) -> List[Dict]:
    """
    Pick the most on-topic sentences across the top chunks. Deterministic:
    score = sum of matched-term weights (IDF when supplied, so a rare word like
    'unreplied' outranks a common one like 'notice'); ties break on
    (chunk order, sentence order). Returns [{text, source(chunk)}].
    """
    qset = set(terms)
    w = weights or {}
    cand = []
    for ci, c in enumerate(chunks):
        for si, s in enumerate(sentences(c["text"])):
            if _BOILERPLATE_RE.search(s):
                continue
            stoks = set(normalize.content_tokens(s))
            matched = qset & stoks
            if not matched:
                continue
            base = sum(w.get(t, 1.0) for t in matched)
            # Down-rank shouty ALL-CAPS heading echoes in favour of real prose.
            letters = [ch for ch in s if ch.isalpha()]
            upper_ratio = (sum(ch.isupper() for ch in letters) / len(letters)) if letters else 0
            score = base - (1.5 if upper_ratio > 0.6 else 0.0)
            cand.append((-score, ci, si, s, c))
    cand.sort(key=lambda x: (x[0], x[1], x[2]))
    picked = []
    seen: Set[str] = set()
    for _, _, _, s, c in cand:
        key = s[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append({"text": s, "chunk": c})
        if len(picked) >= limit:
            break
    return picked


def window_around(context: str, needle: str, width: int = 100) -> str:
    """A readable snippet of `context` centred on `needle` (e.g. an amount).

    P2 ledger text extracts as long run-ons; this trims to the phrase that
    actually carries the value so answers stay clean. Verbatim - no rewording.
    """
    ctx = (context or "").strip()
    if not ctx:
        return ""
    idx = ctx.find(needle)
    if idx < 0 or len(ctx) <= width * 2:
        return ctx if len(ctx) <= width * 2 else ctx[: width * 2].rstrip() + " …"
    start = max(0, idx - width)
    end = min(len(ctx), idx + len(needle) + width)
    snip = ctx[start:end].strip()
    if start > 0:
        # begin at a clean word boundary, not mid-token
        cut = snip.find(" ")
        snip = "… " + (snip[cut + 1:] if 0 <= cut < 20 else snip)
    if end < len(ctx):
        snip = snip + " …"
    return snip


def source_line(sources: List[Dict]) -> str:
    """Render a compact, de-duplicated provenance line."""
    seen = []
    for s in sources:
        rid = s.get("report_id")
        page = s.get("page")
        label = rid + ((" p%d" % page) if page else "")
        if label not in seen:
            seen.append(label)
    return "Source: " + "; ".join(seen) if seen else ""


# Identifier noise that leaks in from the interleaved P2 ledger columns and
# makes an answer read like a dump instead of a decision. Stripped verbatim
# (removing masked GSTIN/PAN and gate ticks never changes the accounting sense).
_GSTIN_RE = re.compile(r"\b\d{2}[0-9A-Z]{13}\b")
_PAN_RE = re.compile(r"\b[A-Z]{2}XXX\d{4}[A-Z]\b")
# Invoice / vendor-ref codes that leak from column 1 (AC/0934, RIT/2627/0412,
# PA-2025-118, OPC-1187, SPX-2241). Shaped so statutory refs survive:
# DRC-03/ASMT-11/GSTR-9 have <=2 or 1 trailing digit; Circ. 9/2025 starts numeric.
_INVREF_RE = re.compile(r"\b[A-Z]{2,4}/\d[\dA-Z/]*\b|\b[A-Z]{2,4}-\d{4}(?:-\d+)?\b")


def clean_reason(text: str) -> str:
    t = _GSTIN_RE.sub("", text or "")
    t = _PAN_RE.sub("", t)
    t = _INVREF_RE.sub("", t)
    t = t.replace("✓", "").replace("✗", "")
    t = re.sub(r"·\s*(?:·\s*)+", "· ", t)
    t = re.sub(r"\s+", " ", t).strip(" ·—-")
    return t


def _tidy_citation(c: str) -> str:
    c = re.sub(r"\s+", " ", (c or "")).strip()
    # drop unbalanced trailing ')' from captures like '§206AA)' (keep §17(5)(b)(i))
    while c.count(")") > c.count("("):
        c = c[:-1].rstrip()
    return c


def verdict_card(v: Dict, report_label: str, reason: str = None) -> str:
    """Render a ledger decision the way a CA reads it: verdict → figures →
    statutory basis → action/reason → proof. Built only from clean structured
    fields, so it is precise and never invents anything."""
    who = v.get("item") or v.get("party") or "Ledger item"
    lines = ["%s — %s" % (who, report_label)]

    pill = v.get("verdict_alias") or "—"
    cls = v.get("verdict_class") or "—"
    meaning = v.get("verdict_meaning") or ""
    ca = " · CA review required" if v.get("ca_review") else ""
    lines.append("Verdict: %s (%s%s)%s" % (pill, cls, (" — " + meaning) if meaning else "", ca))

    amts = v.get("amounts") or []
    if amts:
        lines.append("Amount: " + " · ".join(a["raw"] for a in amts[:3]))
    if v.get("citations"):
        cites = [_tidy_citation(c) for c in v["citations"][:6]]
        lines.append("Statutory basis: " + " · ".join(c for c in cites if c))
    if reason:
        r = clean_reason(reason)
        if r:
            lines.append("Reason: " + (r[:210].rstrip() + " …" if len(r) > 210 else r))
    lines.append("Proof: %s · %s p2" % (v.get("proof") or "—", v.get("report_id", "")))
    return "\n".join(lines)


def rupees(value) -> str:
    """Indian-grouped rupees from an integer (e.g. 180000 -> '₹1,80,000')."""
    if value is None:
        return ""
    n = int(value)
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(?<=\d)(?=(\d\d)+$)", ",", head)
        s = head + "," + tail
    return ("-" if n < 0 else "") + "₹" + s
