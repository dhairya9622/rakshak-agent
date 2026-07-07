#!/usr/bin/env python3
"""
Minimal text REPL for the offline agent (handy for manual testing).

    python cli.py                       # interactive
    python cli.py "how much RCM?"       # one-shot
    python cli.py --json "list vendors" # structured output

This is a thin text wrapper around Agent.ask(); it is NOT a frontend. The same
Agent object is what a frontend would call.
"""

import io
import json
import os
import sys

# force UTF-8 stdout so ₹ / § render on any console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rakshak_agent import Agent, SmartAgent, DeepSeekClient


def _run(agent, q, as_json, smart):
    ans = agent.ask(q)
    if as_json:
        print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
    elif smart:
        print(ans.text)
        print("  [tier=%s · difficulty=%s · llm=%s · cost=$%.6f]"
              % (ans.tier, ans.difficulty, ans.llm_used, ans.cost_usd))
    else:
        print(ans.text)
        print("  [intent=%s · confidence=%.2f · in_scope=%s]"
              % (ans.intent, ans.confidence, ans.in_scope))


def main(argv):
    as_json = "--json" in argv
    smart = "--smart" in argv         # enable the LLM escalation layer
    argv = [a for a in argv if a not in ("--json", "--smart")]
    knowledge_dir = "knowledge"

    if smart:
        # DeepSeek as the cheap tier if a key is present; else offline-only.
        cheap = DeepSeekClient(tier="cheap") if os.environ.get("DEEPSEEK_API_KEY") else None
        capable = (DeepSeekClient(name="deepseek-reasoner", tier="capable",
                                  model="deepseek-reasoner", price_in_per_m=0.55,
                                  price_out_per_m=2.19)
                   if os.environ.get("DEEPSEEK_API_KEY") else None)
        agent = SmartAgent.load(knowledge_dir, cheap_llm=cheap, capable_llm=capable)
    else:
        agent = Agent.load(knowledge_dir)

    if argv:
        _run(agent, " ".join(argv), as_json, smart)
        return 0

    print("Rakshak offline agent. Ask about the reports. Ctrl-C / 'quit' to exit.\n")
    print(agent.ask("help").text, "\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in ("quit", "exit", "q"):
            return 0
        if not q:
            continue
        print()
        _run(agent, q, as_json, smart)
        print()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
