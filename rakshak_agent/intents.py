"""
Intent detection.

Classifies a question into one of a small, fixed set of intents. The intent
only decides which knowledge index to prioritise and how to shape the answer;
retrieval always runs underneath, so a mis-detected intent degrades gracefully
rather than failing. Deterministic keyword/regex rules, no learning.
"""

from __future__ import annotations

import re
from typing import List

from . import normalize

# Intent constants
ADVICE = "advice"         # virtual-CA: prioritise / windows / what-to-do / posture
ROLE = "role"             # meta/boundary: "can you decide/sign/represent for me?"
IDENTITY = "identity"     # report id, GSTIN, due date, status, run hash, version...
AMOUNT = "amount"         # how much, value, total, liability, interest...
VERDICT = "verdict"       # decision/outcome for an item or party
COUNT = "count"           # how many
LIST = "list"             # list / which / enumerate
DEFINE = "define"         # what is X / what does X mean
EXPLAIN = "explain"       # why / how / tell me about  (default, extractive)


_IDENTITY_PATTraws = [
    r"\breport (id|number|#)\b", r"\bgstin\b", r"\btan\b", r"\bpan\b",
    r"\bdue date\b", r"\bdeadline\b", r"\bfiling due\b", r"\breply due\b",
    r"\bstatus\b", r"\brun hash\b", r"\bhash\b", r"\bproof\b",
    r"\bgenerated\b", r"\bengine\b", r"\bversion\b",
    r"\bwho (is|are) the (taxpayer|entity|deductor|company)\b",
    r"\bwhich (company|entity|taxpayer)\b", r"\bwhat (period|fy|year)\b",
    r"\bepigraph\b", r"\bglance\b", r"\bbasis\b",
    r"\bwhen (is|was|are)\b.*\b(due|filed|generated|issued)\b",
]
_IDENTITY_RE = [re.compile(p) for p in _IDENTITY_PATTraws]

_AMOUNT_RE = re.compile(
    r"\bhow much\b|\bwhat amount\b|\bamount\b|\bvalue of\b|\btotal\b|\bsum\b|"
    r"\bliability\b|\binterest\b|\bpenalty\b|\bhow many rupees\b|₹|\bcost\b|"
    r"\bpaid\b|\bpayable\b|\bdeposited\b|\bturnover\b")

_COUNT_RE = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b|\bhow much (?=.*\bare\b)")

_LIST_RE = re.compile(
    r"\blist\b|\bwhich\b|\bwhat are the\b|\ball (the )?(vendors|parties|suppliers|"
    r"decisions|verdicts|reports|deductees|paras|items|gates)\b|\benumerate\b|"
    r"\bshow (me )?(all|the list)\b")

_DEFINE_RE = re.compile(
    r"^\s*(what|whats|what's) (is|are|does)\b.*\b(mean|means|stand for)\b|"
    r"\bdefine\b|\bmeaning of\b|\bwhat does\b.*\bmean\b|\bwhat is (a|an|the)?\s*\w+\??$")

_VERDICT_RE = re.compile(
    r"\bverdict\b|\bdecision\b|\boutcome\b|\bwhat happened\b|\bstatus of\b|"
    r"\bclassif\w+\b|\bwas .* (blocked|claimed|reversed|deferred|paid|reconciled|"
    r"explained|admitted|allowed|denied)\b|\bis .* (eligible|blocked|allowed)\b|"
    r"\bhow (is|was) .* (treated|handled|classified)\b")

_EXPLAIN_RE = re.compile(r"\bwhy\b|\bhow\b|\bexplain\b|\btell me about\b|\breason\b|"
                         r"\bwhat about\b|\bdescribe\b|\bwhat happens?\b")


# NOTE: use whole words \b(advise|advice)\b so the vendor name "Pinnacle
# Advisory" ('advisory') never trips the advice intent.
_ADVICE_RE = re.compile(
    r"\bprioriti|what should i\b|what do i do\b|next step|action item|to-?do\b|"
    r"\bupcoming\b|\bdeadlines\b|\bwhat.s due\b|due next\b|"
    r"\bwhat do you think i should\b|\bdeal with\b|\bwhat.?s next\b|"
    r"\bdo (this |next )?(week|month|quarter|fortnight|today)\b|\bnext (week|month|quarter)\b|"
    r"\bmy (exposure|risk|liability|position)\b|total (exposure|risk|liability)\b|"
    r"\bclosing window|open window|window closing|windows?\b.*clos|"
    r"\brecommend|\b(advise|advice)\b|\bwhat to do\b|posture\b|where do i stand\b|"
    r"\bnotice (position|posture|status|stand)\b")

# Meta/boundary questions about the agent's authority (it advises, never rules).
_ROLE_RE = re.compile(
    r"\btake (a |the )?decisions?\b|\bdecide (for|on) me\b|\bon my behalf\b|"
    r"\bmake (the |my )?decisions?\b|\bsign(-| )?off\b|\bsign it off\b|"
    r"\brepresent me\b|\bact for me\b|\bfile for me\b|\bdecide for me\b|"
    r"\bdo you (decide|rule|sign|file|represent)\b|"
    r"\bcan you (decide|rule|sign|file|represent)\b|"
    r"\bare you (a |an )?(ca|chartered accountant|lawyer|auditor|human|bot|ai|robot)\b")


def detect(question: str, has_entity: bool) -> str:
    q = normalize.phrase_fold(question)

    # Boundary/role questions first ("can you decide/sign for me?").
    if _ROLE_RE.search(q):
        return ROLE

    # Virtual-CA advisory cues next (prioritise / windows / posture / follow-ups).
    if _ADVICE_RE.search(q) or (has_entity and re.search(
            r"\bwhat (should|do) i do\b|\bwhat about\b|\bshould i (pay|reverse|contest|claim)\b|"
            r"\bdeal with\b|\bwhat do you think\b", q)):
        return ADVICE

    # Order matters: most specific first.
    if _COUNT_RE.search(q):
        return COUNT
    if _LIST_RE.search(q):
        return LIST
    if any(rx.search(q) for rx in _IDENTITY_RE):
        return IDENTITY
    # A question naming a party and asking about treatment -> verdict.
    if has_entity and (_VERDICT_RE.search(q) or re.search(
            r"\b(happen|treat|handl|classif|verdict|decision|outcome|status)\b", q)):
        return VERDICT
    if _VERDICT_RE.search(q):
        return VERDICT
    if _AMOUNT_RE.search(q):
        return AMOUNT
    if _DEFINE_RE.search(q):
        return DEFINE
    if _EXPLAIN_RE.search(q):
        return EXPLAIN
    return EXPLAIN
