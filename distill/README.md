# Distillation — train a cheaper student from stronger (DeepSeek) outputs

Goal: over time, answer more of the *hard* questions at the *cheap* tier (or
fully offline), so cost keeps falling. We do this by **distillation**: collect
the capable model's answers as training data, then fine-tune a smaller/cheaper
student on them.

Nothing here trains at runtime — it only captures data and documents the loop.

## 1. Capture teacher outputs (runtime, automatic)

Attach a `DistillationLogger` to the `SmartAgent`. Every time the **capable**
tier answers a hard question, the `(system, grounded-context, answer)` triple is
appended to a JSONL dataset in the OpenAI/DeepSeek chat fine-tuning format:

```python
from rakshak_agent import SmartAgent, DeepSeekClient, DistillationLogger

logger = DistillationLogger("distill/teacher_dataset.jsonl", tiers=("capable_llm",))
sa = SmartAgent.load(
    "knowledge",
    cheap_llm=DeepSeekClient(tier="cheap"),
    capable_llm=DeepSeekClient(name="deepseek-reasoner", tier="capable",
                               model="deepseek-reasoner"),
    teacher_logger=logger,
)
# ... serve real traffic ...  logger.count grows with each captured teacher answer
```

Each line looks like:

```json
{"messages":[
   {"role":"system","content":"You are Rakshak Assistant ... INSUFFICIENT_CONTEXT"},
   {"role":"user","content":"CONTEXT:\n[ANN-2425-FY-0012 p2] ...\n\nQUESTION: ..."},
   {"role":"assistant","content":"<teacher answer, grounded + cited>"}],
 "meta":{"question":"...","teacher_model":"deepseek-reasoner","tier":"capable_llm"}}
```

Because the context is already the minimal grounded slice, the student learns to
answer well **from small context** — which is exactly the cheap-inference regime.

## 2. Curate

- De-duplicate (semantic cache already reduces repeats).
- Optionally keep only answers that a verifier accepts (e.g. cite a real report
  id and contain no amount absent from context).
- Aim for coverage across all five report types and question styles.

## 3. Fine-tune the student (offline/vendor step)

Point your fine-tuning job at `teacher_dataset.jsonl`. Two common targets:

- **Fine-tuning for style/behaviour:** fine-tune the cheap model so it matches
  the grounded, cited, concise Rakshak answer style — fewer abstains, so more
  HARD questions resolve at the cheap tier.
- **Distillation to a smaller student:** fine-tune a small local model on the
  DeepSeek teacher answers, then register it as the `cheap_llm` (or an offline
  tier below it). More traffic then avoids the API entirely.

## 4. Re-wire and measure

Swap the improved student in as `cheap_llm`, keep the strong model as
`capable_llm`, and watch `SmartAgent.stats()`:

- `capable_llm_calls` should drop (student handles more).
- `llm_call_rate` and `total_usd` should fall.
- Answer quality (grounding, citations) should hold — verify against
  `tests/` before/after.

This is the flywheel: strong model teaches → cheap student improves → cost per
answered question decreases, while the deterministic layer keeps answering the
bulk for free.
