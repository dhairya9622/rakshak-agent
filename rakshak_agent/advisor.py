"""
Virtual-CA reasoning engine (deterministic, $0).

Reasons over the extracted report data + an injectable clock to produce SHORT,
precise advice: what is due, which statutory windows are closing, what to do
about a party, and the notice position. No LLM, no invention - every figure and
date comes from the reports; only day-counts are computed (and shown).

Answers are intentionally terse: verdict/figure/deadline, nothing else.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from . import clock as _clock
from .responder import clean_reason, rupees

# Statutory-window keyword -> short label. Used to attach a computed countdown
# to the dated obligations already stated in the ledger.
_WINDOW_KEYS = [
    ("§16(4)", "§16(4) credit window"), ("16(4)", "§16(4) credit window"),
    ("R.37A", "R.37A reversal"), ("37A", "R.37A reversal"),
    ("43B(h)", "MSMED §43B(h) disallowance"), ("MSMED", "MSMED §43B(h) disallowance"),
    ("LDC", "LDC cap"), ("§395", "LDC cap"),
]

_MOD_ACTION = {"ITC": "GSTR-3B", "ANN": "GSTR-9/9C", "TDS": "Form 140", "NTC": "ASMT-11 reply"}


def _urgency(days: int) -> str:
    if days < 0:
        return "OVERDUE"
    if days <= 7:
        return "critical"
    if days <= 21:
        return "soon"
    return "ok"


class Advisor:
    def __init__(self, kb, clock: _clock.Clock = None):
        self.kb = kb
        self.clock = clock or _clock.Clock()

    # -- deadlines / priority ------------------------------------------- #

    def deadlines(self) -> Tuple[str, list]:
        today = self.clock.today()
        rows = []
        for r in self.kb.reports:
            for kind, key in (("filing", "filing_due"), ("reply", "reply_due")):
                d = _clock.parse_date(r.get(key) or "")
                if not d:
                    continue
                rows.append(((d - today).days, r["report_id"], kind, r.get(key)))
        rows.sort(key=lambda x: (x[0], x[1]))
        if not rows:
            return "No dated deadlines on record.", []
        out = ["Due (as of %s):" % _clock.fmt(today)]
        src = []
        for days, rid, kind, ds in rows:
            mod = rid.split("-")[0]
            form = _MOD_ACTION.get(mod, kind)
            tag = ("OVERDUE %dd" % -days) if days < 0 else "%dd (%s)" % (days, _urgency(days))
            out.append("• %s — %s · %s" % (form, ds, tag))
            src.append({"report_id": rid, "page": 1})
        return "\n".join(out), src

    # -- closing statutory windows -------------------------------------- #

    def windows(self, party: Optional[str] = None) -> Tuple[str, list]:
        today = self.clock.today()
        found = {}  # (label, party) -> (days, date, report_id)
        for v in self.kb.verdicts:
            if v["module"] == "WIRE":
                continue
            if party and v.get("party") != party:
                continue
            txt = v.get("text", "")
            futures = [d for d in _clock.all_dates(txt) if (d - today).days >= 0]
            if not futures:
                continue
            for key, label in _WINDOW_KEYS:
                if key in txt:
                    # the actionable deadline is the latest "by <date>" in the row
                    # (e.g. "reverse by 30 Nov if 3B unfiled by 30 Sep" -> 30 Nov)
                    d = max(futures)
                    who = v.get("party") or v.get("item") or ""
                    k = (label, who)
                    if k not in found or (d - today).days < found[k][0]:
                        found[k] = ((d - today).days, d, v["report_id"])
                    break
        if not found:
            return ("No dated statutory window is open%s."
                    % ((" for " + party) if party else ""), [])
        items = sorted(found.items(), key=lambda kv: (kv[1][0], kv[0][0]))
        out = ["Closing windows (as of %s):" % _clock.fmt(today)]
        src = []
        for (label, who), (days, d, rid) in items:
            tag = "OVERDUE" if days < 0 else "%dd (%s)" % (days, _urgency(days))
            out.append("• %s%s — %s · %s" % (label, (" · " + who) if who else "",
                                             _clock.fmt(d), tag))
            src.append({"report_id": rid, "page": 2})
        return "\n".join(out), src

    # -- per-party advisory --------------------------------------------- #

    def item_advice(self, party: str) -> Optional[Tuple[str, list]]:
        vs = sorted((v for v in self.kb.verdicts if v.get("party") == party),
                    key=lambda v: v["report_id"])
        if not vs:
            return None
        v = vs[-1]  # most recent module
        cite = (v.get("citations") or [None])[0]
        parts = ["%s — %s (%s)" % (party, v.get("verdict_alias"), v["report_id"].split("-")[0])]
        if cite:
            parts[0] += " · " + re.sub(r"\s+", " ", cite).strip()
        if v.get("ca_review"):
            parts.append("CA review required")
        wtext, wsrc = self.windows(party=party)
        line = " · ".join(parts) + "."
        if wsrc:  # append the nearest window as the action deadline (drop dup party)
            w = wtext.split("\n")[1].lstrip("• ").replace(" · " + party, "")
            line += " Deadline: " + w
        src = [{"report_id": v["report_id"], "page": 2}] + wsrc[:1]
        return line, src

    # -- notice position ------------------------------------------------ #

    def notice_posture(self) -> Optional[Tuple[str, list]]:
        ntc = [v for v in self.kb.verdicts if v["module"] == "NTC"]
        r = next((r for r in self.kb.reports if r["module"] == "NTC"), None)
        if not ntc or not r:
            return None
        today = self.clock.today()
        due = _clock.parse_date(r.get("reply_due") or "")
        days = " (%dd)" % (due - today).days if due else ""
        alleged, breakdown = self._notice_figures(r["report_id"])
        head = ""
        if alleged:
            head = "Alleged %s" % alleged
            head += (": " + breakdown + ". ") if breakdown else ". "
        line = "%s%d paras, reply due %s%s." % (head, len(ntc), r.get("reply_due", "n/a"), days)
        return line, [{"report_id": r["report_id"], "page": 1}]

    def _notice_figures(self, rid):
        """Pull the report's stated alleged / explained / paid / admitted line."""
        alleged = breakdown = ""
        for f in self.kb.facts:
            if f["report_id"] != rid:
                continue
            c = f.get("context", "")
            if not breakdown:
                m = re.search(r"(₹[\d,]+ explained · ₹[\d,]+ (?:already )?paid · "
                              r"₹[\d,]+ admitted)", c)
                if m:
                    breakdown = m.group(1)
            if not alleged:
                m = re.search(r"alleged[^\d₹]{0,10}(₹?[\d,]{5,})", c, re.IGNORECASE)
                if m:
                    alleged = ("₹" + m.group(1).lstrip("₹"))
        return alleged, breakdown

    # -- action items (verdicts needing money action) ------------------- #

    def action_items(self) -> Tuple[str, list]:
        order = {"PAY": 0, "REVERSE": 1, "DENY": 2}
        acts = [v for v in self.kb.verdicts if v.get("verdict_class") in order]
        acts.sort(key=lambda v: (order[v["verdict_class"]], v["report_id"]))
        if not acts:
            return "No open money-action items.", []
        out = ["Action items:"]
        src = []
        for v in acts:
            amt = v.get("amounts") or []
            aval = amt[-1]["raw"] if len(amt) >= 2 else (amt[0]["raw"] if amt else "")
            ca = " · CA" if v.get("ca_review") else ""
            out.append("• %s %s%s — %s (%s)%s" % (
                v["verdict_alias"], v.get("party") or v.get("item"),
                (" ₹" + aval.lstrip("₹")) if aval else "",
                (v.get("citations") or [""])[0], v["report_id"].split("-")[0], ca))
            src.append({"report_id": v["report_id"], "page": 2})
        return "\n".join(out), src
