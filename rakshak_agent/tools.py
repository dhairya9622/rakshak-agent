"""
Agent tools — the deterministic engine exposed as callable functions.

The LLM never invents figures; it CALLS these to get exact, cited data from the
knowledge base (verdicts, deadlines, statutory windows, interest math, retrieval).
Each tool returns compact JSON-serialisable data with report/page provenance.

Tool specs are OpenAI/DeepSeek function-calling format.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import normalize


def tool_specs() -> List[Dict]:
    def fn(name, desc, props=None, required=None):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": props or {},
                           "required": required or []}}}
    return [
        fn("get_deadlines", "Filing/reply deadlines across all reports with day-counts as of today (priority)."),
        fn("get_statutory_windows", "Closing statutory windows (§16(4), R.37A, MSMED §43B(h), LDC) with day-counts.",
           {"party": {"type": "string", "description": "optional vendor/party to filter by"}}),
        fn("get_notice_position", "The ASMT-10 notice posture: alleged vs explained/paid/admitted, paras, reply-due."),
        fn("get_action_items", "Open money-action ledger items (PAY / REVERSE / DENY) with amounts and basis."),
        fn("get_vendor", "Everything on one vendor/party across the reports: verdicts, amounts, citations, reason, proof.",
           {"name": {"type": "string", "description": "vendor/party name, e.g. 'Pinnacle Advisory'"}}, ["name"]),
        fn("find_verdict", "Find the ledger verdict(s) best matching a description (item, section, situation).",
           {"query": {"type": "string"}}, ["query"]),
        fn("get_identity", "Report identity facts: GSTIN, TAN, PAN, entity, run hash, engine version, status, due dates.",
           {"subject": {"type": "string", "description": "e.g. 'gstin', 'tan', 'run hash', 'filing due', 'status'"}}, ["subject"]),
        fn("search_reports", "Full-text retrieval over the reports; returns the most relevant passages with report/page.",
           {"query": {"type": "string"}}, ["query"]),
        fn("list_vendors", "List every vendor/party tracked across the reports."),
        fn("list_reports", "List the five reports (id, module, period)."),
        fn("compute_interest", "Deterministic interest: base x annual_rate_pct% x days/365. Use for §50/§398 interest.",
           {"base": {"type": "number"}, "annual_rate_pct": {"type": "number"}, "days": {"type": "integer"}},
           ["base", "annual_rate_pct", "days"]),
    ]


class ToolKit:
    """Dispatches a tool call against the deterministic Agent (kb/index/advisor)."""

    def __init__(self, agent):
        self.agent = agent
        self.kb = agent.kb
        self.index = agent.index
        self.advisor = agent.advisor

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            handler = getattr(self, "_t_" + name, None)
            if handler is None:
                return {"error": "unknown tool: %s" % name}
            return handler(args or {})
        except Exception as exc:  # never crash the loop on a bad tool call
            return {"error": "%s: %s" % (name, exc)}

    # -- advisory tools (real-time) ------------------------------------- #

    def _wrap(self, res):
        if not res:
            return {"result": "none"}
        text, sources = res
        return {"summary": text, "sources": sources}

    def _t_get_deadlines(self, a):
        return self._wrap(self.advisor.deadlines())

    def _t_get_statutory_windows(self, a):
        return self._wrap(self.advisor.windows(party=a.get("party")))

    def _t_get_notice_position(self, a):
        return self._wrap(self.advisor.notice_posture())

    def _t_get_action_items(self, a):
        return self._wrap(self.advisor.action_items())

    # -- ledger / entity tools ------------------------------------------ #

    def _t_get_vendor(self, a):
        name = a.get("name", "")
        ent = self.index.find_entity(name) or self.index.find_entity("about " + name)
        if not ent:
            # fuzzy: match on first word
            for e in self.kb.entities:
                if name.split()[0].lower() in e["name"].lower():
                    ent = e
                    break
        if not ent:
            return {"error": "no vendor matching %r" % name,
                    "known": [e["name"] for e in self.kb.entities]}
        rows = [v for v in self.kb.verdicts if v.get("party") == ent["name"]]
        return {"vendor": ent["name"], "verdicts": [{
            "report_id": v["report_id"], "verdict": v.get("verdict_alias"),
            "class": v.get("verdict_class"), "ca_review": v.get("ca_review"),
            "amounts": [x["raw"] for x in v.get("amounts", [])],
            "citations": v.get("citations", []), "proof": v.get("proof"),
            "reason": self.agent._verdict_reason(v)} for v in rows]}

    def _t_find_verdict(self, a):
        terms = normalize.query_terms(a.get("query", ""))
        vs = self.index.rank_verdicts(terms, [], top_k=3)
        return {"verdicts": [{
            "report_id": v["report_id"], "item": v.get("item") or v.get("party"),
            "verdict": v.get("verdict_alias"), "class": v.get("verdict_class"),
            "amounts": [x["raw"] for x in v.get("amounts", [])],
            "citations": v.get("citations", []), "proof": v.get("proof"),
            "reason": self.agent._verdict_reason(v)} for _, v in vs]}

    def _t_get_identity(self, a):
        subj = (a.get("subject") or "").lower()
        hits = [f for f in self.kb.facts if f["kind"] == "identity"
                and subj in (f["subject"].lower() + " " + str(f["value"]).lower())]
        if not hits:
            hits = [f for f in self.kb.facts if f["kind"] == "identity"
                    and any(t in f["subject"].lower() for t in subj.split())]
        return {"facts": [{"subject": f["subject"], "value": f["value"],
                           "report_id": f["report_id"], "page": f["page"]}
                          for f in hits[:12]]}

    def _t_search_reports(self, a):
        terms = normalize.query_terms(a.get("query", ""))
        ch = self.index.rank_chunks(terms, [], top_k=4)
        return {"passages": [{"report_id": c["report_id"], "page": c["page"],
                              "text": c["text"][:500]} for _, c in ch]}

    def _t_list_vendors(self, a):
        return {"vendors": [e["name"] for e in self.kb.entities]}

    def _t_list_reports(self, a):
        return {"reports": [{"report_id": r["report_id"],
                             "module": r.get("module_label", r["module"]),
                             "period": r.get("period")} for r in self.kb.reports]}

    def _t_compute_interest(self, a):
        base = float(a["base"]); rate = float(a["annual_rate_pct"]); days = int(a["days"])
        interest = round(base * (rate / 100.0) * days / 365.0)
        return {"interest": interest,
                "formula": "%s x %s%% x %d/365 = %d" % (
                    format(int(base), ","), rate, days, interest)}
