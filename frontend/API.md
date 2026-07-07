# Rakshak Virtual-CA Agent — API

**Base URL:** `https://rakshak-agent.onrender.com`
**Auth:** none (public). **CORS:** open (`Access-Control-Allow-Origin: *`).
**Content-Type:** `application/json; charset=utf-8` (UTF-8 — responses contain ₹, §).
**Note:** free host sleeps after ~15 min idle; the first request may take ~50s to wake.

Closed-loop assistant: answers only from 5 fixed compliance reports for Meridian
Components Pvt Ltd. Off-topic questions are refused (never call a model).

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | service info |
| GET | `/health` | readiness + KB stats |
| GET | `/suggested` | the 4 preloaded questions |
| POST | `/ask` | ask a question → answer |
| OPTIONS | `*` | CORS preflight → `204` |

---

### GET /health
```bash
curl https://rakshak-agent.onrender.com/health
```
```json
{ "ok": true, "llm_enabled": true, "reports": 5, "chunks": 111,
  "verdicts": 25, "facts": 367, "entities": 10 }
```
`llm_enabled` = whether the DeepSeek escalation tier is configured on the server.

### GET /suggested
```bash
curl https://rakshak-agent.onrender.com/suggested
```
```json
{ "questions": [
  "What should I prioritise before the deadlines?",
  "Which statutory windows are closing?",
  "What should I do about Pinnacle Advisory?",
  "What is the notice position?" ] }
```

### POST /ask
**Request body:** `{ "question": string, "context"?: { "last_entity"?: string } }`

`context` is optional and enables **multi-turn follow-ups**. Echo the previous
answer's `topic` back as `context.last_entity`, so a follow-up like
*"advise on this"* resolves to the last subject instead of being refused. If the
follow-up has no resolvable subject, the agent asks a one-line clarifier (it does
NOT refuse).
```bash
curl -X POST https://rakshak-agent.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the GSTIN?"}'
```
**Response:** an `Answer` object (see schema below).
```json
{ "text": "Gstin: 27XXXXX1234X1Z5 (consistent across all reports).\nSource: ANN-2425-FY-0012 p1",
  "intent": "identity", "tier": "deterministic", "difficulty": "easy",
  "in_scope": true, "confidence": 3.0, "llm_used": false, "cached": null,
  "model": null, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0,
  "prompt_cache_hit": false,
  "sources": [{"report_id": "ANN-2425-FY-0012", "page": 1}],
  "data": {"items": [{"subject": "GSTIN", "value": "27XXXXX1234X1Z5", "scope": "all"}]} }
```

JavaScript:
```js
const res = await fetch(`${API_URL}/ask`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question }),
});
const answer = await res.json();
```

---

## `Answer` schema (POST /ask response)

| field | type | notes |
|---|---|---|
| `text` | string | the answer; **may contain `\n`** — render as line breaks |
| `intent` | enum | `identity` \| `amount` \| `verdict` \| `count` \| `list` \| `define` \| `explain` \| `advice` \| `llm` \| `out_of_scope` \| `capabilities` \| `empty` |
| `tier` | enum | `deterministic` \| `response_cache` \| `semantic_cache` \| `cheap_llm` \| `capable_llm` \| `refused` |
| `difficulty` | enum | `easy` \| `medium` \| `hard` \| `out_of_scope` |
| `in_scope` | boolean | `false` only for refusals |
| `confidence` | number | ≥ 0 (0 for refusals) |
| `llm_used` | boolean | `true` only when a model was actually called |
| `cached` | string \| null | `null` \| `"response"` \| `"semantic"` |
| `model` | string \| null | e.g. `"deepseek-chat"` / `"deepseek-reasoner"` when `llm_used` |
| `cost_usd` | number | USD spent on this request (0 unless `llm_used`) |
| `tokens_in` | number | prompt tokens (0 unless `llm_used`) |
| `tokens_out` | number | completion tokens |
| `prompt_cache_hit` | boolean | provider prompt-cache discount applied |
| `topic` | string \| null | resolved subject of this answer — **echo back as `context.last_entity`** on the next turn |
| `sources` | array | provenance: `[{ "report_id": string, "page": number }]` |
| `data` | object | intent-specific structured payload (see below) |

**`report_id` values:** `ANN-2425-FY-0012`, `ITC-2627-MAY-0049`,
`NTC-2425-JUL-0003`, `TDS-2627-Q1-0031`, `WIRE-MERIDIAN-0001`.

**`data` by intent (optional, for rich rendering):**
- `identity` → `{ items: [{ subject, value, scope|report_id }] }`
- `verdict` (party) → `{ entity, mentions: [{ report_id, verdict_alias, verdict_class, amount, citations }] }`
- `verdict` (single) → `{ verdict: { verdict_alias, verdict_class, party, item, primary_amount, citations, proof } }`
- `amount` → `{ amounts: [{ value, value_number, report_id, page }] }`
- `count` → `{ count, of }` · `list` → `{ list: string[], of }`
- `advice` → `{ advisory: true }` · `llm` → `{ escalated: true }`

### TypeScript type
```ts
interface Source { report_id: string; page: number; }
interface Answer {
  text: string;
  intent: "identity"|"amount"|"verdict"|"count"|"list"|"define"|"explain"|"advice"|"llm"|"out_of_scope"|"capabilities"|"empty";
  tier: "deterministic"|"response_cache"|"semantic_cache"|"cheap_llm"|"capable_llm"|"refused";
  difficulty: "easy"|"medium"|"hard"|"out_of_scope";
  in_scope: boolean;
  confidence: number;
  llm_used: boolean;
  cached: "response"|"semantic"|null;
  model: string|null;
  cost_usd: number;
  tokens_in: number;
  tokens_out: number;
  prompt_cache_hit: boolean;
  topic: string | null;   // echo back as context.last_entity next turn
  sources: Source[];
  data: Record<string, unknown>;
}
```

---

## Sample responses by type

**Advisory (real-time, deterministic):**
```json
{ "text": "Closing windows (as of 07 Jul 2026):\n• R.37A reversal · Pinnacle Advisory LLP — 30 Nov 2026 · 146d (ok)\n• MSMED §43B(h) disallowance · Annapurna Caterers — 31 Mar 2027 · 267d (ok)",
  "intent": "advice", "tier": "deterministic", "llm_used": false, "cost_usd": 0.0 }
```

**LLM-answered (hard question, grounded + cited):**
```json
{ "text": "The RCM was paid via DRC-03 before filing to voluntarily avoid penalty exposure … [ANN-2425-FY-0012 p3], [NTC-2425-JUL-0003 p2]",
  "intent": "llm", "tier": "cheap_llm", "llm_used": true, "model": "deepseek-chat",
  "cost_usd": 0.000224, "tokens_in": 638, "tokens_out": 90, "prompt_cache_hit": true }
```

**Out of scope (closed loop):**
```json
{ "text": "Out of scope. I only cover the Meridian Rakshak reports (ITC, GSTR-9/9C, TDS, notice).",
  "intent": "out_of_scope", "tier": "refused", "in_scope": false, "cost_usd": 0.0, "sources": [] }
```

---

## Errors

| Status | Body | When |
|---|---|---|
| 400 | `{ "error": "question is required" }` | missing/empty `question` |
| 400 | `{ "error": "invalid JSON body" }` | body isn't valid JSON |
| 404 | `{ "error": "not found" }` | unknown path |

All errors are JSON with the same CORS headers.

## Notes for integrators
- Stateless: no sessions/history; each `/ask` is independent.
- Idempotent-ish: repeated identical questions hit the server cache (`tier: response_cache`, `cost_usd: 0`).
- No rate limit yet — add one (and lock CORS to your domain) before public launch.
- The DeepSeek key lives only on the server; never call DeepSeek from the browser.
