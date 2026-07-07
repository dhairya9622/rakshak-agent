"""
Query normalization, tokenization and domain-aware synonym expansion.

This module holds the ONLY interpretive layer in the agent: it maps how a user
phrases a question onto the vocabulary the reports actually use (GST/TDS/notice
shorthand). It never invents facts - it only rewrites query words so retrieval
can find the right report content. Fully deterministic (pure string ops).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Set


STOPWORDS: Set[str] = set("""
a an the of to in on for and or is are was were be been being with by at as from into per
this that these those it its their his her your you we our us me my i they them he she
not no nor so such up out off over under about above below between within
do does did done has have had having will would can could shall should may might must
what when where why how who whom whose which whether there here then than too very just
please tell show give find explain describe want need know about regarding concerning
report reports pdf document documents file files
""".split())

# Multi-word phrase -> canonical token(s). Applied to the lowercased raw string
# BEFORE tokenizing, so multi-word domain terms collapse to report vocabulary.
PHRASE_SYNONYMS: Dict[str, str] = {
    "input tax credit": "itc credit",
    "reverse charge mechanism": "rcm reverse charge",
    "reverse charge": "rcm reverse charge",
    "annual return": "annual gstr9 gstr-9",
    "annual reconciliation": "annual gstr9 9c reconciliation",
    "tax deducted at source": "tds",
    "due date": "due deadline",
    "deadline": "due deadline",
    "filing due": "filing due deadline",
    "reply due": "reply due deadline",
    "run hash": "run hash",
    "how much": "amount",
    "what amount": "amount",
    "how many": "count",
    "number of": "count number",
    "goods and services tax": "gst",
    "show cause notice": "scn notice",
    "credit note": "cn credit note",
    "blocked credit": "blocked block 17(5)",
    "goods in transit": "transit defer",
    "late fee": "late fee",
    "interest": "interest",
    "vendor score": "vendor score",
    "vendor risk": "vendor risk score",
    "lower deduction certificate": "ldc",
    "self invoice": "self-invoice self invoice",
    "self-invoice": "self-invoice self invoice",
    "gst identification number": "gstin",
    "gst identification": "gstin",
    "identification number": "gstin",
    "gst number": "gstin",
    "gst id": "gstin",
    "tax deduction account number": "tan",
    "permanent account number": "pan",
    "report number": "report id",
    "report id": "report id",
}

# Single-token synonyms / abbreviations -> extra tokens added to the query.
TOKEN_SYNONYMS: Dict[str, List[str]] = {
    "itc": ["credit"],
    "credit": ["itc"],
    "rcm": ["reverse", "charge"],
    "supplier": ["vendor", "party", "deductee", "counterparty"],
    "vendor": ["supplier", "party", "deductee"],
    "party": ["vendor", "supplier"],
    "deductee": ["vendor", "party"],
    "counterparty": ["vendor", "supplier"],
    "notice": ["asmt", "asmt-10", "scrutiny", "drc"],
    "asmt": ["notice"],
    "annual": ["gstr9", "gstr-9", "9c"],
    "gstr9": ["annual"],
    "monthly": ["itc", "3b", "gstr-3b"],
    "quarterly": ["tds", "form", "140"],
    "deadline": ["due", "filing", "reply"],
    "verdict": ["decision", "outcome", "classification"],
    "decision": ["verdict", "outcome"],
    "outcome": ["verdict", "decision"],
    "taxpayer": ["entity", "registered", "person", "deductor"],
    "entity": ["taxpayer", "company"],
    "amount": ["value", "rupees", "sum"],
    "penalty": ["penalty", "271h"],
    "payment": ["paid", "pay", "drc-03", "challan"],
    "paid": ["payment", "drc-03"],
    "hash": ["proof", "sha256"],
    "proof": ["hash"],
    "gate": ["gates", "check"],
    "section": ["§"],
}

# Module trigger words -> module code. Used to bias retrieval to a report.
MODULE_TRIGGERS: Dict[str, str] = {
    "itc": "ITC", "input tax credit": "ITC", "gstr-3b": "ITC", "gstr3b": "ITC",
    "monthly": "ITC", "3b": "ITC", "ims": "ITC",
    "annual": "ANN", "gstr-9": "ANN", "gstr9": "ANN", "9c": "ANN",
    "reconciliation": "ANN", "reconcile": "ANN",
    "tds": "TDS", "26as": "TDS", "form 140": "TDS", "26q": "TDS",
    "deductee": "TDS", "deductor": "TDS", "challan": "TDS", "quarterly": "TDS",
    "notice": "NTC", "asmt": "NTC", "asmt-10": "NTC", "asmt-11": "NTC",
    "scrutiny": "NTC", "drc-01": "NTC", "para": "NTC", "reply": "NTC",
    "wire": "WIRE", "two years": "WIRE", "two-year": "WIRE", "agents talk": "WIRE",
    "timeline": "WIRE", "thread": "WIRE",
}


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def fold(text: str) -> str:
    """Lowercase + strip accents, keep ascii-ish content."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def _light_stem(tok: str) -> str:
    """Very light, deterministic plural/verb stemming (no external libs)."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("ses"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def phrase_fold(text: str) -> str:
    """Lowercase + collapse multi-word domain phrases to report vocabulary.

    Shared by intent detection and identity-subject matching so a paraphrase
    like 'gst identification number' resolves to 'gstin'.
    """
    folded = fold(text)
    for phrase, repl in PHRASE_SYNONYMS.items():
        if phrase in folded:
            folded = folded.replace(phrase, " " + repl + " ")
    return folded


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(phrase_fold(text))


def content_tokens(text: str) -> List[str]:
    """Tokens minus stopwords, lightly stemmed. Order preserved, de-duped."""
    out: List[str] = []
    seen: Set[str] = set()
    for t in tokenize(text):
        if t in STOPWORDS or len(t) < 2:
            continue
        s = _light_stem(t)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def expand(tokens: List[str]) -> List[str]:
    """Add domain synonyms so a query matches report vocabulary."""
    out: List[str] = list(tokens)
    seen: Set[str] = set(tokens)
    for t in tokens:
        for syn in TOKEN_SYNONYMS.get(t, ()):
            s = _light_stem(syn)
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def query_terms(text: str) -> List[str]:
    """Full pipeline: content tokens + synonym expansion (deterministic order)."""
    return expand(content_tokens(text))


def detect_modules(text: str) -> List[str]:
    """Which report module(s) the question points at, in stable order."""
    folded = fold(text)
    hits: List[str] = []
    for trigger, module in MODULE_TRIGGERS.items():
        if trigger in folded and module not in hits:
            hits.append(module)
    return hits
