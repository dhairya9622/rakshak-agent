"""
The agent.

Deterministic, offline question-answering over the preprocessed Rakshak
knowledge base. No LLM, no network, no randomness. Public surface:

    agent = Agent.load("knowledge")
    ans = agent.ask("How much was the RCM liability?")
    print(ans.text)          # clean text response
    ans.sources              # structured provenance (frontend-friendly)
    ans.to_dict()            # everything, JSON-serialisable

Design principle: answers are assembled only from stored facts/verdicts and
verbatim report sentences, so the agent stays strictly inside report knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import intents, normalize, responder
from .index import Index
from .knowledge import KnowledgeBase

# Confidence below which we decline instead of guessing.
_MIN_KNOWN_FRACTION = 0.30
_MIN_SIGNAL = 0.5


@dataclass
class Answer:
    text: str
    intent: str
    in_scope: bool = True
    confidence: float = 0.0
    sources: List[Dict] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "intent": self.intent,
            "in_scope": self.in_scope,
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "data": self.data,
        }


# Which identity subject(s) a question is asking about.
_IDENTITY_SUBJECTS = [
    (("gstin",), "GSTIN"),
    (("tan",), "TAN"),
    (("pan",), "PAN"),
    (("report id", "report #", "report no"), "report id"),
    (("run", "hash", "sha256"), "run hash"),
    (("status",), "status"),
    (("reply", "due"), "reply due date"),
    (("filing", "due", "deadline", "file"), "filing due date"),
    (("generated", "when", "date"), "generated"),
    (("version", "engine"), "engine version"),
    (("period", "fy", "year", "quarter"), "period"),
    (("entity", "taxpayer", "company", "who", "deductor", "registered"), "entity"),
    (("module", "kind", "type"), "module"),
    (("epigraph",), "epigraph"),
    (("glance",), "glance"),
    (("basis",), "basis"),
    (("decision", "desk"), "decisions to desk"),
]

_HELP_RE = re.compile(
    r"\b(help|what can you (do|answer)|capabilit|how do i use|who are you|what do you know)\b")
_GREET_RE = re.compile(r"^\s*(hi|hello|hey|yo|greetings)\b")

# A follow-up that points back at the previous topic ("advise on this",
# "what about it", "deal with them") — deliberately NOT matching "this week".
_ANAPHORA_RE = re.compile(
    r"\b(on|about|with|for|regarding) (this|that|it|them|those|these)\b|"
    r"\b(this|that) one\b|\b(deal with|advise on|advice on|handle) (this|that|it|them)\b")


def is_anaphoric(question: str) -> bool:
    return bool(_ANAPHORA_RE.search(normalize.fold(question)))


class Agent:
    def __init__(self, kb: KnowledgeBase, clock=None):
        self.kb = kb
        self.index = Index(kb)
        from .advisor import Advisor
        self.advisor = Advisor(kb, clock=clock)

    @classmethod
    def load(cls, knowledge_dir: str = "knowledge", clock=None) -> "Agent":
        return cls(KnowledgeBase.load(knowledge_dir), clock=clock)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ask(self, question: str, context: dict = None) -> Answer:
        q = (question or "").strip()
        if not q:
            return Answer("Please ask a question about the Rakshak reports.",
                          "empty", in_scope=True, confidence=0.0)

        folded = normalize.phrase_fold(q)
        if _GREET_RE.search(folded) or _HELP_RE.search(folded):
            return self._capabilities()

        raw_terms = normalize.content_tokens(q)
        terms = normalize.expand(raw_terms)
        modules = normalize.detect_modules(q)
        entity = self.index.find_entity(q)
        # Resolve a follow-up ("advise on this") to the prior turn's topic.
        if entity is None and is_anaphoric(q) and context and context.get("last_entity"):
            entity = self.kb.entity_by_name.get(context["last_entity"])
        entity_name = entity["name"] if entity else None
        intent = intents.detect(q, has_entity=bool(entity))

        # Scope is judged on informative (alphabetic) terms only: a bare number
        # like a year can coincide with a citation and must not imply relevance.
        alpha_terms = [t for t in raw_terms if any(c.isalpha() for c in t)]
        known = self.index.known_fraction(alpha_terms) if alpha_terms else 0.0

        # Route by intent; each router returns an Answer or None (no confident hit).
        ans: Optional[Answer] = None
        if intent == intents.ROLE:
            ans = self._role_answer()
        elif intent == intents.ADVICE:
            ans = self._answer_advice(q, folded, entity)
        elif intent == intents.COUNT:
            ans = self._answer_count(q, folded, modules)
        elif intent == intents.LIST:
            ans = self._answer_list(q, folded, modules)
        elif intent == intents.IDENTITY:
            ans = self._answer_identity(q, folded, terms, modules)
        elif intent == intents.VERDICT:
            ans = self._answer_verdict(q, terms, modules, entity)
        elif intent == intents.AMOUNT:
            ans = self._answer_amount(q, terms, modules)
        elif intent == intents.DEFINE:
            ans = self._answer_define(q, folded, terms, modules)

        # Fallback / EXPLAIN: extractive retrieval.
        if ans is None:
            ans = self._answer_explain(q, terms, modules, entity)

        if ans is None:
            return self._out_of_scope(q)

        # Out-of-scope guard. Structured answers (identity/amount/count/list/
        # verdict) only fire when they matched specific stored data, so they are
        # trusted. Extractive answers (explain/define) can latch onto coincidental
        # matches, so we require real alphabetic overlap with the knowledge.
        if ans.intent in (intents.EXPLAIN, intents.DEFINE) and not entity and not modules:
            if known == 0.0 or (ans.confidence < _MIN_SIGNAL and known < _MIN_KNOWN_FRACTION):
                return self._out_of_scope(q)
        # Tag the resolved topic so a frontend can echo it back as context
        # for the next follow-up ("advise on this").
        if entity_name and isinstance(ans.data, dict) and "topic" not in ans.data:
            ans.data["topic"] = entity_name
        return ans

    def _role_answer(self) -> Answer:
        text = ("I'm a decision-aid over the five reports: I surface verdicts, figures, "
                "deadlines and their statutory basis, each cited to report and page. "
                "I don't take decisions, sign off, or represent you — the ruling and "
                "filing rest with you and your CA.")
        return Answer(text, intents.ROLE, in_scope=True, confidence=3.0,
                      sources=[], data={"role": True})

    # ------------------------------------------------------------------ #
    # Intent routers
    # ------------------------------------------------------------------ #

    def _answer_advice(self, q, folded, entity) -> Optional[Answer]:
        """Virtual-CA advisory routing (deterministic, real-time via the clock)."""
        adv = self.advisor
        has_time_or_task = re.search(
            r"prioriti|deadline|due|week|month|quarter|today|next|do\b|step|to-?do", folded)
        if entity:
            res = adv.item_advice(entity["name"])
        elif re.search(r"window|§?16\(4\)|37a|msmed|43b|ldc|clos", folded):
            res = adv.windows()
        elif re.search(r"exposure|risk|liabilit|action|owe|reverse", folded):
            res = adv.action_items()
        elif re.search(r"notice|asmt|posture|drc|scrutiny", folded):
            res = adv.notice_posture()
        elif is_anaphoric(q) or not has_time_or_task:
            # a follow-up with no resolvable subject ("advise on this") -> ask,
            # don't refuse and don't dump an unrelated chunk.
            return Answer(
                "About which item? Name a vendor, report, or para — e.g. "
                "\"What should I do about Pinnacle Advisory?\"",
                intents.ADVICE, in_scope=True, confidence=1.0, sources=[],
                data={"clarify": True})
        else:  # prioritise / deadlines / what's next
            res = adv.deadlines()
        if not res:
            return None
        text, sources = res
        return Answer(text, intents.ADVICE, confidence=3.0, sources=sources,
                      data={"advisory": True})

    def _answer_identity(self, q, folded, terms, modules) -> Optional[Answer]:
        subjects = [subj for keys, subj in _IDENTITY_SUBJECTS
                    if any(k in folded for k in keys)]
        # de-dup keep order
        subjects = list(dict.fromkeys(subjects))
        if not subjects:
            return None

        facts = [f for f in self.kb.facts if f["kind"] == "identity"
                 and f["subject"] in subjects
                 and (not modules or f["module"] in modules)]
        if not facts:
            return None

        lines: List[str] = []
        sources: List[Dict] = []
        data_items = []
        for subj in subjects:
            group = [f for f in facts if f["subject"] == subj]
            if not group:
                continue
            values = list(dict.fromkeys(f["value"] for f in group))
            if len(values) == 1 and len(group) > 1 and not modules:
                lines.append("%s: %s (consistent across all reports)."
                             % (subj.capitalize(), values[0]))
                sources.append({"report_id": group[0]["report_id"], "page": group[0]["page"]})
                data_items.append({"subject": subj, "value": values[0], "scope": "all"})
            else:
                for f in group:
                    lines.append("%s (%s): %s." % (subj.capitalize(),
                                 self.kb.report_label(f["report_id"]), f["value"]))
                    sources.append({"report_id": f["report_id"], "page": f["page"]})
                    data_items.append({"subject": subj, "value": f["value"],
                                       "report_id": f["report_id"]})

        text = "\n".join(lines)
        if sources:
            text += "\n" + responder.source_line(sources)
        return Answer(text, intents.IDENTITY, confidence=3.0,
                      sources=sources, data={"items": data_items})

    def _answer_amount(self, q, terms, modules) -> Optional[Answer]:
        facts = self.index.rank_facts(terms, modules, kinds=("amount",), top_k=4)
        if not facts:
            # maybe the amount lives on a verdict row
            vs = self.index.rank_verdicts(terms, modules, top_k=1)
            if vs and vs[0][1].get("primary_amount") is not None:
                v = vs[0][1]
                text = "%s — %s (%s). \n%s" % (
                    responder.rupees(v["primary_amount"]),
                    v.get("item") or v.get("party") or "ledger item",
                    self.kb.report_label(v["report_id"]),
                    responder.source_line([{"report_id": v["report_id"], "page": 2}]))
                return Answer(text, intents.AMOUNT, confidence=vs[0][0],
                              sources=[{"report_id": v["report_id"], "page": 2}],
                              data={"amount": v["primary_amount"]})
            return None

        top_score = facts[0][0]
        # keep facts close to the best (deterministic band) and cap at 3
        best = [(s, f) for s, f in facts if s >= top_score * 0.6][:3]
        lines, sources, data_items = [], [], []
        for s, f in best:
            snippet = responder.clean_reason(
                responder.window_around(f["context"], f["value"], width=90))
            lines.append("%s — %s" % (f["value"], snippet.rstrip(".") + "."))
            sources.append({"report_id": f["report_id"], "page": f["page"]})
            data_items.append({"value": f["value"], "value_number": f.get("value_number"),
                               "report_id": f["report_id"], "page": f["page"]})
        text = "\n".join(lines) + "\n" + responder.source_line(sources)
        return Answer(text, intents.AMOUNT, confidence=top_score,
                      sources=sources, data={"amounts": data_items})

    def _answer_verdict(self, q, terms, modules, entity) -> Optional[Answer]:
        if entity:
            return self._answer_entity(entity, terms)
        vs = self.index.rank_verdicts(terms, modules, top_k=3)
        if not vs:
            return None
        score, v = vs[0]
        card = responder.verdict_card(v, self.kb.report_label(v["report_id"]),
                                      reason=self._verdict_reason(v))
        src = [{"report_id": v["report_id"], "page": 2}]
        return Answer(card, intents.VERDICT, confidence=score, sources=src,
                      data={"verdict": {k: v.get(k) for k in
                            ("verdict_id", "verdict_alias", "verdict_class",
                             "party", "item", "primary_amount", "citations", "proof")}})

    def _answer_entity(self, entity, terms) -> Answer:
        name = entity["name"]
        # The ledger verdict rows for this party carry the reason + citations.
        vrows = sorted((v for v in self.kb.verdicts if v.get("party") == name),
                       key=lambda v: v["report_id"])
        lines, sources, data_items = [], [], []
        for v in vrows:
            amt = (" · " + responder.rupees(v["primary_amount"])) if v.get("primary_amount") else ""
            ca = " +CA REVIEW" if v.get("ca_review") else ""
            reason = self._verdict_reason(v)
            cites = [responder._tidy_citation(c) for c in v.get("citations", [])[:3]]
            cite = (" [%s]" % " · ".join(c for c in cites if c)) if cites else ""
            lines.append("• %s: %s (%s)%s%s — %s%s" % (
                self.kb.report_label(v["report_id"]), v.get("verdict_alias") or "—",
                v.get("verdict_class") or "—", amt, ca, reason, cite))
            sources.append({"report_id": v["report_id"], "page": 2})
            data_items.append({"report_id": v["report_id"],
                               "verdict_alias": v.get("verdict_alias"),
                               "verdict_class": v.get("verdict_class"),
                               "amount": v.get("primary_amount"),
                               "citations": v.get("citations")})
        # Narrative-only mentions (e.g. the WIRE two-year timeline) add colour.
        for m in sorted(entity["mentions"], key=lambda m: (m["report_id"], m["page"])):
            if m.get("verdict_class") is None and m["report_id"].startswith("WIRE"):
                snip = responder.clean_reason(m["snippet"])
                # the WIRE timeline reads "MON 20xx ... {name} ..."; start there
                mm = re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+20\d\d",
                               snip)
                if mm:
                    snip = snip[mm.start():]
                lines.append("• %s (two-year wire): %s" % (
                    self.kb.report_label(m["report_id"]), snip.rstrip(".") + "."))
                sources.append({"report_id": m["report_id"], "page": m["page"]})
                break
        if not lines:
            m = sorted(entity["mentions"], key=lambda m: (m["report_id"], m["page"]))[0]
            lines.append("• %s: %s" % (self.kb.report_label(m["report_id"]), m["snippet"]))
            sources.append({"report_id": m["report_id"], "page": m["page"]})

        header = "%s across the reports:" % name
        text = header + "\n" + "\n".join(lines) + "\n" + responder.source_line(sources)
        return Answer(text, intents.VERDICT, confidence=5.0, sources=sources,
                      data={"entity": name, "mentions": data_items})

    def _sentence_weights(self, q, terms, primary=None):
        """IDF weights for sentence selection, with synonym-expanded terms
        discounted so a user's exact rare word (e.g. 'unreplied') outranks a
        looser synonym match (e.g. 'scrutiny')."""
        prim = set(primary if primary is not None else normalize.content_tokens(q))
        return {t: self.index._idf(t) * (1.0 if t in prim else 0.4) for t in set(terms)}

    @staticmethod
    def _verdict_reason(v, width: int = 190) -> str:
        """Verbatim reason prose from a ledger row: the text right after the pill,
        trimmed before the statutory-citation tail."""
        text = v.get("text") or ""
        alias = v.get("verdict_alias") or ""
        up = text.upper()
        idx = up.find(alias.upper()) if alias else -1
        rest = text[idx + len(alias):] if idx >= 0 else text
        # cut at the citation/proof tail
        rest = re.split(r"\s(?:§|proof:|R\.\d|Rule\s\d|Circ\.|Notif\.|Annexure)\b",
                        rest, maxsplit=1)[0]
        rest = responder.clean_reason(rest)     # drop GSTIN/PAN/invoice/tick noise
        # Prefer the first full sentence: the P2 columns interleave, so text
        # after the first period is usually bleed from other columns.
        first = re.split(r"(?<=[.;])\s", rest, maxsplit=1)[0]
        if len(first) >= 30:
            rest = first
        return (rest[:width].rstrip() + " …") if len(rest) > width else rest

    def _answer_count(self, q, folded, modules) -> Optional[Answer]:
        target = self._count_target(folded)
        if target is None:
            return None
        kind, count, detail, sources = target
        text = "%d %s%s" % (count, kind, ("." if not detail else ": " + detail + "."))
        if sources:
            text += "\n" + responder.source_line(sources)
        return Answer(text, intents.COUNT, confidence=3.0, sources=sources,
                      data={"count": count, "of": kind})

    def _all_report_sources(self):
        return [{"report_id": r["report_id"], "page": 1} for r in self.kb.reports]

    def _count_target(self, folded):
        mods = normalize.detect_modules(folded)
        # Order matters: a specific noun (decisions/vendors/gates) beats the
        # generic word "report" (which appears in phrases like "the TDS report").
        if re.search(r"\bvendors?\b|\bparties\b|\bsuppliers?\b|\bdeductees?\b|"
                               r"\bcounterpart", folded):
            names = [e["name"] for e in self.kb.entities]
            return ("vendors/parties tracked", len(self.kb.entities), "; ".join(names),
                    self._all_report_sources())
        if re.search(r"\bdecisions?\b|\bverdicts?\b|\bledger\b|\bparas?\b|"
                               r"\bitems?\b|\blines? .*desk\b|\bflag", folded):
            vs = [v for v in self.kb.verdicts if not mods or v["module"] in mods]
            if mods and len(mods) == 1:
                rid = self.kb.reports_by_module.get(mods[0], [None])[0]
                vs = [v for v in self.kb.verdicts if v["module"] == mods[0]]
                labels = ["%s (%s)" % (v.get("item") or v.get("party"), v["verdict_alias"])
                          for v in vs]
                return ("decisions in %s" % self.kb.report_label(rid), len(vs),
                        "; ".join(l for l in labels if l), [{"report_id": rid, "page": 2}])
            return ("decisions across all reports", len(self.kb.verdicts), "", [])
        if re.search(r"\bgates?\b", folded):
            gs = [f for f in self.kb.facts if f["kind"] == "gate"
                  and (not mods or f["module"] in mods)]
            names = [f["value"] for f in gs]
            return ("gates", len(gs), "; ".join(names), [])
        if re.search(r"\bentit", folded):
            return ("entities tracked", len(self.kb.entities),
                    "; ".join(e["name"] for e in self.kb.entities), [])
        if re.search(r"\breports?\b|\bmodules?\b", folded):
            names = [self.kb.report_label(r["report_id"]) for r in self.kb.reports]
            return ("reports", len(self.kb.reports), "; ".join(names),
                    self._all_report_sources())
        return None

    # Nouns that merely name a collection to enumerate (not a search predicate).
    _LIST_NOUNS = frozenset((
        "vendor", "party", "supplier", "deductee", "counterparty", "list",
        "report", "module", "decision", "verdict", "ledger", "para", "item",
        "gate", "entity", "tracked", "show", "enumerate", "there", "all",
        "give", "name",
    ))

    def _answer_list(self, q, folded, modules) -> Optional[Answer]:
        # "which vendor has an inoperative PAN" carries a predicate -> it is a
        # search, not an enumeration. Detect discriminating terms and, if present,
        # answer the specific hit instead of dumping the whole collection.
        content = normalize.content_tokens(q)
        extra = [t for t in content if t not in self._LIST_NOUNS]
        if extra:
            # rank on the discriminating predicate only ('inoperative', 'pan'),
            # not the collection noun ('vendor') which would drag in synonyms.
            terms = normalize.expand(extra)
            entity = self.index.find_entity(q)
            if entity:
                return self._answer_entity(entity, terms)
            vs = self.index.rank_verdicts(terms, modules, top_k=1)
            if vs and vs[0][0] >= 2.0:
                return self._answer_verdict(q, terms, modules, None)
            ch = self.index.rank_chunks(terms, modules, top_k=3)
            if ch and ch[0][0] >= 2.0:
                picks = responder.select_sentences([c for _, c in ch], terms, limit=2,
                                                   weights=self._sentence_weights(q, terms, primary=extra))
                if picks:
                    src = [{"report_id": p["chunk"]["report_id"],
                            "page": p["chunk"]["page"]} for p in picks]
                    txt = " ".join(p["text"] for p in picks) + "\n" + responder.source_line(src)
                    return Answer(txt, intents.EXPLAIN, confidence=ch[0][0], sources=src)

        # Otherwise: a genuine enumeration.
        target = self._count_target(folded) or self._count_target("decisions " + folded)
        if target is None:
            return None
        kind, count, detail, sources = target
        items = [d.strip() for d in detail.split(";") if d.strip()] if detail else []
        if not items:
            return None
        text = "%s (%d):\n" % (kind.capitalize(), count) + \
               "\n".join("• " + it for it in items)
        if sources:
            text += "\n" + responder.source_line(sources)
        return Answer(text, intents.LIST, confidence=3.0, sources=sources,
                      data={"list": items, "of": kind})

    def _answer_define(self, q, folded, terms, modules) -> Optional[Answer]:
        # 1) verdict-taxonomy term?
        term = self._define_term(folded)
        if term:
            for v in self.kb.verdicts:
                if (v.get("verdict_alias") or "").lower() == term or \
                   (v.get("verdict_class") or "").lower() == term:
                    text = "%s is a %s verdict (%s). Example: %s in %s.\n%s" % (
                        v["verdict_alias"], v["verdict_class"], v["verdict_meaning"],
                        v.get("item") or v.get("party"),
                        self.kb.report_label(v["report_id"]),
                        responder.source_line([{"report_id": v["report_id"], "page": 2}]))
                    return Answer(text, intents.DEFINE, confidence=3.0,
                                  sources=[{"report_id": v["report_id"], "page": 2}],
                                  data={"term": v["verdict_alias"],
                                        "class": v["verdict_class"]})
        # 2) fall back to extractive definition from report text
        chunks = self.index.rank_chunks(terms, modules, top_k=4)
        if chunks and chunks[0][0] >= 1.0:
            picks = responder.select_sentences([c for _, c in chunks], terms, limit=2,
                                               weights=self._sentence_weights(q, terms))
            if picks:
                sources = [{"report_id": p["chunk"]["report_id"],
                            "page": p["chunk"]["page"]} for p in picks]
                text = " ".join(p["text"] for p in picks) + "\n" + \
                       responder.source_line(sources)
                return Answer(text, intents.DEFINE, confidence=chunks[0][0],
                              sources=sources, data={})
        return None

    def _define_term(self, folded):
        m = re.search(
            r"(?:what (?:is|are|does)|define|meaning of)\s+(?:an?\s+|the\s+)?"
            r"([a-z0-9\-\+ ]{2,30}?)(?:\s+mean| stand for|\?|$)", folded)
        if m:
            return m.group(1).strip()
        return None

    def _answer_explain(self, q, terms, modules, entity) -> Optional[Answer]:
        if entity:
            # entity-centric explanation combines its ledger rows + narrative
            return self._answer_entity(entity, terms)
        chunks = self.index.rank_chunks(terms, modules, top_k=6)
        if not chunks or chunks[0][0] <= 0:
            return None
        picks = responder.select_sentences([c for _, c in chunks], terms, limit=3,
                                           weights=self._sentence_weights(q, terms))
        if not picks:
            # no sentence overlap; surface the single best chunk verbatim (trimmed)
            c = chunks[0][1]
            snippet = c["text"][:400]
            src = [{"report_id": c["report_id"], "page": c["page"]}]
            return Answer(snippet + "\n" + responder.source_line(src),
                          intents.EXPLAIN, confidence=chunks[0][0], sources=src)
        sources = [{"report_id": p["chunk"]["report_id"], "page": p["chunk"]["page"]}
                   for p in picks]
        text = " ".join(p["text"] for p in picks) + "\n" + responder.source_line(sources)
        return Answer(text, intents.EXPLAIN, confidence=chunks[0][0],
                      sources=sources, data={})

    # ------------------------------------------------------------------ #
    # Non-answers
    # ------------------------------------------------------------------ #

    def _out_of_scope(self, q) -> Answer:
        # Closed-loop: only the Meridian Rakshak reports. Terse refusal.
        text = "Out of scope. I only cover the Meridian Rakshak reports (ITC, GSTR-9/9C, TDS, notice)."
        return Answer(text, "out_of_scope", in_scope=False, confidence=0.0)

    def _capabilities(self) -> Answer:
        s = self.kb.stats()
        lines = [
            "I am an offline agent that answers questions strictly from the "
            "Rakshak Systems reports for %s." % (
                self.kb.reports[0].get("entity") if self.kb.reports else "this entity"),
            "Loaded knowledge: %d reports, %d ledger decisions, %d facts, "
            "%d vendors/parties." % (s["reports"], s["verdicts"], s["facts"], s["entities"]),
            "Reports:",
        ]
        for r in self.kb.reports:
            lines.append("• %s — %s %s" % (r["report_id"], r.get("module_label", r["module"]),
                                           ("· " + r["period"]) if r.get("period") else ""))
        lines += [
            "You can ask, for example:",
            "• \"What is the verdict on Pinnacle Advisory?\"",
            "• \"How much was the RCM liability?\"",
            "• \"When is the TDS statement due?\"",
            "• \"List the vendors\" / \"How many decisions are in the notice reply?\"",
            "• \"Why was Annapurna Caterers blocked?\"",
        ]
        return Answer("\n".join(lines), "capabilities", confidence=1.0)
