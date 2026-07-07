"""
Distillation / fine-tuning data capture.

Closes the cost loop: whenever the EXPENSIVE (capable) model answers a hard
question, we log the exact (system, grounded-context, answer) triple as a
training example. Those teacher outputs later fine-tune a cheaper student
(or the cheap tier itself) so more questions get answered at the low tier -
pushing cost down over time.

Output is JSONL in the OpenAI/DeepSeek chat fine-tuning format:
  {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}],
   "meta":{...}}

Purely a recorder - deterministic, offline, no training happens here. See
distill/README.md for the DeepSeek distillation workflow.
"""

from __future__ import annotations

import json
import os
from typing import Optional


class DistillationLogger:
    def __init__(self, path: str = "distill/teacher_dataset.jsonl",
                 tiers=("capable_llm",)):
        self.path = path
        self.tiers = set(tiers)          # which tiers count as "teacher" outputs
        self.count = 0
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def record(self, question: str, system: str, user: str, answer: str,
               model: str, tier: str, sources=None) -> bool:
        if tier not in self.tiers or not answer:
            return False
        row = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
            "meta": {"question": question, "teacher_model": model,
                     "tier": tier, "sources": sources or []},
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1
        return True


def load_dataset(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
