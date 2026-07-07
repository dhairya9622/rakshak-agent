# Rakshak Offline Knowledge Agent

A deterministic, offline question-answering agent over the Rakshak Systems
compliance reports (ITC / GSTR-9 annual / TDS / Notice / two-year wire), with an
optional, aggressively cost-optimised LLM escalation layer on top.

**Design contract:** the offline deterministic agent is the foundation and
answers everything it can at **zero LLM cost**. A model is called only for
genuinely hard, in-domain questions — cheapest tier first, smallest context,
with caching. Truly out-of-scope questions are refused for free (never a paid
call). Answers are assembled only from stored facts/verdicts and verbatim report
sentences, so the offline agent cannot hallucinate.

---

## 1. Install & build the knowledge base

```bash
python -m pip install -r requirements.txt          # pdfplumber (preprocessing only)
python preprocess.py --input "C:/Users/DHAIRYA/Downloads/files (2)" --out knowledge
```

`preprocess.py` runs locally on the PDF folder and writes a deterministic
knowledge base (`knowledge/*.json`): `reports`, `chunks`, `verdicts`, `facts`,
`entities`, plus a `manifest` with per-report sanity checks. Same PDFs in →
byte-identical JSON out.

## 2. Ask questions

Offline, zero-cost:

```python
from rakshak_agent import Agent
agent = Agent.load("knowledge")
print(agent.ask("How much was the RCM liability?").text)
```

With the cost-optimised LLM layer (offline still answers most things for free):

```python
from rakshak_agent import SmartAgent, DeepSeekClient
sa = SmartAgent.load(
    "knowledge",
    cheap_llm=DeepSeekClient(tier="cheap"),                       # DEEPSEEK_API_KEY
    capable_llm=DeepSeekClient(name="deepseek-reasoner", tier="capable",
                               model="deepseek-reasoner"),
)
ans = sa.ask("Compare how Pinnacle Advisory is treated across the reports")
print(ans.text, ans.tier, ans.llm_used, ans.cost_usd)
print(sa.stats())     # cost / cache / routing dashboard
```

CLI:

```bash
python cli.py "what is the verdict on Pinnacle Advisory?"     # offline
python cli.py --smart "compare Pinnacle across the reports"   # escalation layer
python cli.py --json "how much RCM?"                          # structured output
```

## 3. Test

```bash
python -m unittest discover -s tests        # 78 tests, stdlib only (no pytest)
```

- `tests/test_agent.py` — accuracy, determinism, out-of-scope, edge cases (56).
- `tests/test_llm_layer.py` — routing, caching, cost, fallback, quality (22).

---

## Architecture (module map)

| Concern | Module | What it does |
|---|---|---|
| Deterministic preprocessing / symbolic layer | `preprocess.py` | PDFs → chunks + facts + verdicts + entities (grammar-aware, deterministic) |
| Knowledge base retrieval (RAG) | `rakshak_agent/index.py` | BM25-lite over chunks + scored fact/verdict/entity lookup |
| Indexing / embedding pipeline | `rakshak_agent/embeddings.py` | offline hashed embeddings for semantic cache + retrieval |
| Cheap agent for easy questions | `rakshak_agent/engine.py` | the deterministic agent (`Agent.ask → Answer`) |
| Difficulty classification | `rakshak_agent/classifier.py` | easy / medium / hard / out-of-scope at the start of every request |
| Model routing / LLM cascade | `rakshak_agent/router.py`, `llm.py` | cheapest-first cascade, escalate only on abstain; cost meter |
| Minimal context selection | `rakshak_agent/context.py` | smallest grounded slice (hybrid BM25 + embeddings) |
| Semantic + response caching | `rakshak_agent/cache.py` | exact + paraphrase reuse; stable prompt prefix for prompt caching |
| Orchestrator | `rakshak_agent/smart_agent.py` | wires it all together (`SmartAgent.ask → SmartAnswer`) |
| Distillation from stronger outputs | `rakshak_agent/distillation.py`, `distill/` | logs teacher (capable-model) answers for fine-tuning a cheaper student |

## Cost-optimisation flow (per request)

```
question
  → exact response cache            hit? return         (zero cost)
  → deterministic agent (always; free) + routing signals
  → semantic (paraphrase) cache     hit? return         (zero cost, topic-guarded)
  → classify difficulty
      EASY          → deterministic answer               (zero cost)
      OUT_OF_SCOPE  → refuse                              (zero cost, NO model call)
      MEDIUM        → cheap model over minimal context    (low cost) — or offline fallback
      HARD          → cheap → capable cascade             (escalate only if cheap abstains)
  → cache result (response + semantic) for future reuse
```

**Cost levers:** (1) most traffic never leaves the offline tier; (2) external
questions are refused free; (3) the LLM sees only a tiny grounded context, never
whole reports; (4) a stable system prefix maximises provider prompt-cache hits;
(5) the cascade tries the cheapest model first and escalates only on an explicit
`INSUFFICIENT_CONTEXT` abstain; (6) exact + paraphrase caches reuse prior
answers; (7) teacher outputs can be distilled into a cheaper student over time.

`SmartAgent.stats()` reports `llm_call_rate`, `zero_cost_fraction`, cache hits,
tokens, and `total_usd` for accountability.

## Guarantees

- **Offline:** the agent uses only the standard library and the local knowledge
  base. The LLM tier is optional and pluggable.
- **Deterministic:** preprocessing is byte-identical; the offline agent returns
  identical answers for identical questions (case/whitespace-insensitive).
- **No hallucination (offline):** every amount/verdict/date printed exists in the
  knowledge base (enforced by tests). The LLM tier is constrained to the provided
  context and must reply `INSUFFICIENT_CONTEXT` when it cannot answer.
- **Frontend-ready:** `Answer`/`SmartAnswer` are plain dataclasses with
  `.to_dict()` (text + intent/tier + confidence + sources + cost).
