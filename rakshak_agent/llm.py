"""
LLM provider abstraction + model cascade + cost model.

The agent is provider-agnostic. Everything the router needs is behind the
LLMClient contract, so you can wire DeepSeek (cheap), a stronger model
(capable), a local model, or a mock - without changing routing logic.

Cost discipline lives here too:
  * every call reports token usage and an estimated USD cost
  * ModelCascade tries the CHEAPEST client first and only escalates when the
    cheap model explicitly abstains (ABSTAIN sentinel) - never speculatively.

Prices are illustrative defaults and fully configurable per client; nothing
here asserts a vendor's current price list.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# The cheap model must emit exactly this when the context cannot answer the
# question, so the cascade knows to escalate instead of returning a guess.
ABSTAIN = "INSUFFICIENT_CONTEXT"


def estimate_tokens(text: str) -> int:
    """Cheap, stable token estimate (~4 chars/token). Good enough for routing
    and cost accounting without a tokenizer dependency."""
    return max(1, len((text or "")) // 4)


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    prompt_cache_hit: bool = False
    abstained: bool = False
    meta: Dict = field(default_factory=dict)


class LLMClient:
    """Base client. Subclass and implement _complete()."""

    def __init__(self, name: str, tier: str = "cheap",
                 price_in_per_m: float = 0.0, price_out_per_m: float = 0.0,
                 price_in_cached_per_m: Optional[float] = None):
        self.name = name
        self.tier = tier
        self.price_in = price_in_per_m
        self.price_out = price_out_per_m
        # Cached (prompt-cache-hit) input tokens are usually far cheaper.
        self.price_in_cached = (price_in_cached_per_m
                                if price_in_cached_per_m is not None
                                else price_in_per_m * 0.25)
        self.calls = 0

    def cost(self, tokens_in: int, tokens_out: int, cached_in: int = 0) -> float:
        fresh_in = max(0, tokens_in - cached_in)
        return (fresh_in / 1_000_000.0) * self.price_in \
            + (cached_in / 1_000_000.0) * self.price_in_cached \
            + (tokens_out / 1_000_000.0) * self.price_out

    def complete(self, system: str, user: str, cached_prefix_tokens: int = 0) -> LLMResponse:
        self.calls += 1
        text = self._complete(system, user)
        t_in = estimate_tokens(system) + estimate_tokens(user)
        t_out = estimate_tokens(text)
        resp = LLMResponse(
            text=text.strip(), model=self.name,
            tokens_in=t_in, tokens_out=t_out,
            cost_usd=self.cost(t_in, t_out, cached_in=cached_prefix_tokens),
            prompt_cache_hit=cached_prefix_tokens > 0,
            abstained=text.strip() == ABSTAIN,
        )
        return resp

    def _complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Test / offline clients
# --------------------------------------------------------------------------- #

class MockLLMClient(LLMClient):
    """Deterministic client driven by a handler(system, user) -> str."""

    def __init__(self, name="mock", tier="cheap", handler: Callable = None, **kw):
        super().__init__(name, tier, **kw)
        self._handler = handler or (lambda s, u: ABSTAIN)

    def _complete(self, system, user):
        return self._handler(system, user)


class ScriptedLLMClient(LLMClient):
    """Returns a canned answer when a trigger substring is in the user prompt,
    else ABSTAINs. Handy for cascade / routing tests."""

    def __init__(self, name="scripted", tier="cheap", script: Dict[str, str] = None, **kw):
        super().__init__(name, tier, **kw)
        self.script = script or {}

    def _complete(self, system, user):
        low = user.lower()
        for trigger, answer in self.script.items():
            if trigger.lower() in low:
                return answer
        return ABSTAIN


# --------------------------------------------------------------------------- #
# Real adapter (optional; needs network + API key). Degrades gracefully.
# --------------------------------------------------------------------------- #

class DeepSeekClient(LLMClient):
    """
    OpenAI-compatible DeepSeek adapter (stdlib urllib, no extra deps).

    DeepSeek performs automatic server-side context (prompt) caching keyed on a
    shared prefix, so we always send the SAME stable system prompt first to
    maximise cache hits. If no key / no network, complete() raises and the
    cascade falls through to the next client (or the offline fallback).
    """

    API_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, name="deepseek-chat", tier="cheap",
                 api_key_env="DEEPSEEK_API_KEY", model="deepseek-chat",
                 price_in_per_m=0.27, price_out_per_m=1.10,
                 price_in_cached_per_m=0.07, timeout=30, temperature=0.0):
        super().__init__(name, tier, price_in_per_m, price_out_per_m, price_in_cached_per_m)
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def _complete(self, system, user):  # pragma: no cover - network path
        import urllib.request
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError("DeepSeek API key not set (%s)" % self.api_key_env)
        payload = {
            "model": self.model,
            "temperature": self.temperature,  # 0 -> as deterministic as possible
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            self.API_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Cascade: cheapest first, escalate only on abstain
# --------------------------------------------------------------------------- #

@dataclass
class ChatTurn:
    """One assistant turn from a chat/tool-calling model."""
    content: str = ""
    tool_calls: list = field(default_factory=list)   # [{id, name, arguments(dict)}]
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""
    prompt_cache_hit: bool = False


class ChatClient:
    """Chat + tool-calling contract. Subclass and implement _chat()."""

    def __init__(self, name, price_in_per_m=0.0, price_out_per_m=0.0,
                 price_in_cached_per_m=None):
        self.name = name
        self.price_in = price_in_per_m
        self.price_out = price_out_per_m
        self.price_in_cached = (price_in_cached_per_m if price_in_cached_per_m is not None
                                else price_in_per_m * 0.25)
        self.calls = 0

    def cost(self, t_in, t_out, cached_in=0):
        fresh = max(0, t_in - cached_in)
        return (fresh / 1e6) * self.price_in + (cached_in / 1e6) * self.price_in_cached \
            + (t_out / 1e6) * self.price_out

    def chat(self, messages, tools=None) -> ChatTurn:
        self.calls += 1
        return self._chat(messages, tools)

    def _chat(self, messages, tools):  # pragma: no cover
        raise NotImplementedError


class DeepSeekChatClient(ChatClient):
    """DeepSeek chat completions with tool-calling (stdlib urllib)."""

    API_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, name="deepseek-chat", model="deepseek-chat",
                 api_key_env="DEEPSEEK_API_KEY", price_in_per_m=0.27,
                 price_out_per_m=1.10, price_in_cached_per_m=0.07,
                 timeout=60, temperature=0.0, max_tokens=700):
        super().__init__(name, price_in_per_m, price_out_per_m, price_in_cached_per_m)
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _chat(self, messages, tools):  # pragma: no cover - network
        import urllib.request
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError("DeepSeek API key not set (%s)" % self.api_key_env)
        payload = {"model": self.model, "temperature": self.temperature,
                   "max_tokens": self.max_tokens, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            self.API_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        data = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        t_in = usage.get("prompt_tokens", 0)
        t_out = usage.get("completion_tokens", 0)
        cached = usage.get("prompt_cache_hit_tokens", 0)
        calls = []
        for tc in (msg.get("tool_calls") or []):
            try:
                a = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:
                a = {}
            calls.append({"id": tc["id"], "name": tc["function"]["name"], "arguments": a})
        return ChatTurn(content=msg.get("content") or "", tool_calls=calls,
                        tokens_in=t_in, tokens_out=t_out,
                        cost_usd=self.cost(t_in, t_out, cached),
                        model=self.model, prompt_cache_hit=cached > 0)


class MockChatClient(ChatClient):
    """Scripted chat client for tests. `script` is a list of ChatTurn (or dicts)
    returned in order; each call pops the next. Lets tests drive a tool loop
    deterministically."""

    def __init__(self, name="mock-chat", script=None, price_in_per_m=0.27,
                 price_out_per_m=1.10, **kw):
        super().__init__(name, price_in_per_m, price_out_per_m, **kw)
        self.script = list(script or [])
        self.seen_messages = []       # captured for assertions
        self.seen_tools = None

    def _chat(self, messages, tools):
        self.seen_messages = messages
        self.seen_tools = tools
        turn = self.script.pop(0) if self.script else ChatTurn(content="(no script)")
        if isinstance(turn, dict):
            turn = ChatTurn(**turn)
        t_in = sum(estimate_tokens(str(m.get("content") or "")) for m in messages)
        t_out = estimate_tokens(turn.content)
        turn.tokens_in = turn.tokens_in or t_in
        turn.tokens_out = turn.tokens_out or t_out
        turn.cost_usd = self.cost(turn.tokens_in, turn.tokens_out)
        turn.model = self.name
        return turn


class ModelCascade:
    """Ordered clients (cheapest -> most capable). Tries each in turn; a client
    that ABSTAINs or errors hands off to the next. Records every attempt so the
    router can account for total cost."""

    def __init__(self, clients: List[LLMClient]):
        self.clients = clients

    def run(self, system: str, user: str, cached_prefix_tokens: int = 0):
        attempts: List[LLMResponse] = []
        for client in self.clients:
            try:
                resp = client.complete(system, user, cached_prefix_tokens)
            except Exception as exc:  # network/key failure -> escalate/fallback
                attempts.append(LLMResponse(text="", model=client.name,
                                            abstained=True, meta={"error": str(exc)}))
                continue
            attempts.append(resp)
            if not resp.abstained and resp.text:
                return resp, attempts
        return None, attempts  # nobody could answer
