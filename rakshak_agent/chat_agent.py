"""
ChatAgent — the conversational AI agent (memory + reasoning + tool use).

This is the "proper agent": a capable model holds the full conversation and
reasons over the reports, but every fact it states comes from a TOOL call into
the deterministic engine (exact verdicts, figures, deadlines, interest, search).
So it is genuinely intelligent and conversational, yet stays grounded and cited.

  agent = ChatAgent(deterministic_agent, chat_client)
  reply = agent.chat([{ "role": "user", "content": "how should I handle Pinnacle?" }])
  reply.text, reply.sources, reply.cost_usd, reply.tools_used

Trade-offs (accepted for this mode): not $0, not deterministic, needs the model.
Grounding via tools keeps hallucination low, not zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import clock as _clock
from .tools import ToolKit, tool_specs


@dataclass
class ChatResponse:
    text: str
    model: Optional[str] = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_iterations: int = 0
    tools_used: List[str] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    prompt_cache_hit: bool = False
    fell_back: bool = False        # true if the model was unavailable

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["cost_usd"] = round(self.cost_usd, 6)
        return d


class ChatAgent:
    def __init__(self, agent, chat_client=None, clock=None,
                 max_tool_iters: int = 6):
        self.agent = agent                      # deterministic Agent (kb/index/advisor)
        self.client = chat_client               # ChatClient or None (offline)
        self.toolkit = ToolKit(agent)
        self.clock = clock or _clock.Clock()
        self.max_tool_iters = max_tool_iters
        self._specs = tool_specs()

    # ------------------------------------------------------------------ #

    def _system_prompt(self) -> str:
        reps = "\n".join("- %s: %s%s" % (
            r["report_id"], r.get("module_label", r["module"]),
            (" (" + r["period"] + ")") if r.get("period") else "")
            for r in self.agent.kb.reports)
        entity = self.agent.kb.reports[0].get("entity", "the registered person") \
            if self.agent.kb.reports else "the registered person"
        return (
            "You are Rakshak, a virtual-CA assistant for %s. Today is %s.\n"
            "You have exactly these five compliance reports and NOTHING else:\n%s\n\n"
            "How you work:\n"
            "- Answer ONLY from these reports. For ANY figure, verdict, date, "
            "citation, deadline or vendor fact, CALL A TOOL to get the exact value "
            "— never state a number or section from memory.\n"
            "- Reason across reports, connect patterns, and give clear, practical "
            "guidance a Chartered Accountant can act on.\n"
            "- Always cite the report id (and page when given) for facts you use.\n"
            "- You are a decision-aid: you advise, compute and draft, but you do "
            "not take decisions, sign off, or represent anyone — the ruling and "
            "filing rest with the CA and %s.\n"
            "- If a question is outside these five reports, say so briefly; do not "
            "use outside knowledge.\n"
            "- Be concise, precise, and cite. Indian rupee grouping (e.g. "
            "₹1,80,000)."
            % (entity, _clock.fmt(self.clock.today()), reps, entity))

    def chat(self, messages: List[Dict], context: dict = None) -> ChatResponse:
        """messages = full conversation [{role:'user'|'assistant', content}]."""
        user_msgs = [m for m in (messages or []) if m.get("content")]
        if not user_msgs:
            return ChatResponse("Ask me about the five Rakshak reports.")

        if self.client is None:
            return self._offline_fallback(user_msgs)

        convo = [{"role": "system", "content": self._system_prompt()}] + [
            {"role": m["role"], "content": m["content"]} for m in user_msgs]

        cost = t_in = t_out = 0
        tools_used: List[str] = []
        sources: List[Dict] = []
        cache_hit = False
        model = None

        for i in range(self.max_tool_iters):
            try:
                turn = self.client.chat(convo, tools=self._specs)
            except Exception:
                # model/network failure mid-conversation -> graceful offline answer
                fb = self._offline_fallback(user_msgs)
                fb.cost_usd += cost
                fb.fell_back = True
                return fb
            cost += turn.cost_usd
            t_in += turn.tokens_in
            t_out += turn.tokens_out
            cache_hit = cache_hit or turn.prompt_cache_hit
            model = turn.model

            if not turn.tool_calls:
                return ChatResponse(
                    text=(turn.content or "").strip(), model=model, cost_usd=cost,
                    tokens_in=t_in, tokens_out=t_out, tool_iterations=i,
                    tools_used=tools_used, sources=_dedupe(sources),
                    prompt_cache_hit=cache_hit)

            # record the assistant tool-call turn, then answer each tool
            convo.append({"role": "assistant", "content": turn.content or "",
                          "tool_calls": [{"id": tc["id"], "type": "function",
                                          "function": {"name": tc["name"],
                                                       "arguments": json.dumps(tc["arguments"])}}
                                         for tc in turn.tool_calls]})
            for tc in turn.tool_calls:
                result = self.toolkit.dispatch(tc["name"], tc["arguments"])
                tools_used.append(tc["name"])
                sources.extend(_collect_sources(result))
                convo.append({"role": "tool", "tool_call_id": tc["id"],
                              "content": json.dumps(result, ensure_ascii=False)})

        # hit the tool-iteration cap: ask the model for a final answer, no tools
        try:
            turn = self.client.chat(convo + [{"role": "user",
                    "content": "Give your final answer now from the tool results above."}],
                    tools=None)
            cost += turn.cost_usd
            return ChatResponse(text=(turn.content or "").strip(), model=model,
                                cost_usd=cost, tokens_in=t_in, tokens_out=t_out,
                                tool_iterations=self.max_tool_iters,
                                tools_used=tools_used, sources=_dedupe(sources),
                                prompt_cache_hit=cache_hit)
        except Exception:
            fb = self._offline_fallback(user_msgs)
            fb.cost_usd += cost
            fb.fell_back = True
            return fb

    def _offline_fallback(self, user_msgs) -> ChatResponse:
        """No model available: answer the latest turn with the deterministic
        agent so the endpoint still works (single-shot, not conversational)."""
        last = user_msgs[-1]["content"]
        ctx = {"last_entity": _last_entity(user_msgs)}
        ans = self.agent.ask(last, context=ctx)
        return ChatResponse(text=ans.text, model=None, cost_usd=0.0,
                            sources=ans.sources, fell_back=True)


def _collect_sources(result: Dict) -> List[Dict]:
    out = []
    if not isinstance(result, dict):
        return out
    for s in result.get("sources", []):
        out.append(s)
    for key in ("verdicts", "facts", "passages"):
        for row in result.get(key, []) or []:
            if isinstance(row, dict) and row.get("report_id"):
                out.append({"report_id": row["report_id"], "page": row.get("page", 2)})
    return out


def _dedupe(sources: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for s in sources:
        k = (s.get("report_id"), s.get("page"))
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _last_entity(user_msgs) -> Optional[str]:
    return None
