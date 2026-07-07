# Claude Design Brief — Rakshak Virtual-CA Chatbot

Upload **this file** to Claude Design to generate the chat interface. It fully
specifies the UI and the backend API it wires to. Build a single-page chat app.

---

## 1. What to build

A **closed-loop virtual-CA chatbot** for one company's compliance reports
(Meridian Components Pvt Ltd — GST ITC, GSTR-9 annual, TDS, and a notice/DRC
pack). The assistant:

- Answers **only** from these reports (a closed loop). Anything off-topic gets a
  one-line refusal — do **not** build general chat, tools, or web features.
- Gives **short, precise** answers: figures, verdicts, deadlines. No filler.
- Is a **virtual CA**: it reasons over real-time deadlines and statutory windows.

It is a chat surface with a **preloaded suggestion row** of 4 flagship questions.

---

## 2. Backend API (already built & hosted — wire the UI to these)

The agent runs as an **external hosted API** (Render / Hugging Face Space).
Read the base URL from an env var — do NOT hard-code it:
`API_URL = process.env.NEXT_PUBLIC_API_URL` (Next.js) or `import.meta.env.VITE_API_URL`
(Vite). Dev default `http://127.0.0.1:8000`. The API sends open CORS headers, so
the browser calls it cross-origin directly. The DeepSeek key stays on the backend —
never call DeepSeek from the frontend.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{ ok, llm_enabled, reports, verdicts, ... }` |
| GET | `/suggested` | — | `{ questions: string[4] }` (the preloaded chips) |
| POST | `/ask` | `{ "question": "..." }` | a **SmartAnswer** (below) |

**SmartAnswer** (POST /ask response) — render `text`; use the rest for badges:
```json
{
  "text": "string (may contain \n — render as line breaks)",
  "intent": "identity|amount|verdict|advice|explain|out_of_scope|...",
  "tier": "deterministic|response_cache|semantic_cache|cheap_llm|capable_llm|refused",
  "in_scope": true,
  "confidence": 3.0,
  "llm_used": false,
  "cached": null,                 // or "response" | "semantic"
  "model": null,                  // e.g. "deepseek-chat" when llm_used
  "cost_usd": 0.0,
  "sources": [{ "report_id": "ANN-2425-FY-0012", "page": 1 }],
  "data": {}
}
```

Fetch `/suggested` on load to populate the chips.

**Use `POST /chat` for the conversation** (a real AI agent with memory + tools —
it reasons across reports and cites exact figures). Keep a `messages[]` array in
state: on send, append `{role:"user", content}`, POST `{ messages }` (the whole
array), render `reply.text`, then append `{role:"assistant", content: reply.text}`.
Show small footers from the response: `model`, `cost_usd`, `tools_used`, and the
`sources` chips. (`/ask` still exists for single-shot $0 widget lookups.)

---

## 3. Real sample responses (use these to design the message bubbles)

**Advisory (deadlines) — terse, monospace list:**
```json
{ "text": "Due (as of 07 Jul 2026):\n• GSTR-9/9C — 31 Dec 2025 · OVERDUE 188d\n• GSTR-3B — 20 Jun 2026 · OVERDUE 17d\n• ASMT-11 reply — 22 Jul 2026 · 15d (soon)\n• Form 140 — 31 Jul 2026 · 24d (ok)",
  "intent": "advice", "tier": "deterministic", "llm_used": false, "cost_usd": 0.0,
  "sources": [{"report_id":"ANN-2425-FY-0012","page":1}] }
```

**Fact:**
```json
{ "text": "Gstin: 27XXXXX1234X1Z5 (consistent across all reports).\nSource: ANN-2425-FY-0012 p1",
  "intent": "identity", "tier": "deterministic", "cost_usd": 0.0,
  "sources": [{"report_id":"ANN-2425-FY-0012","page":1}] }
```

**Decision card (verdict) — labelled lines:**
```json
{ "text": "Orbit Packaging Co — TDS-2627-Q1-0031 (FY 2026-27)\nVerdict: FIX (REPAIRED — auto-repaired, then allowed) · CA review required\nAmount: 4,00,000 · 80,000 · ₹72,000\nStatutory basis: NS01 · Circ. 9/2025 · §206AA\nProof: 5f0d…a318 · TDS-2627-Q1-0031 p2",
  "intent": "verdict", "tier": "deterministic", "cost_usd": 0.0 }
```

**LLM-answered (hard question) — grounded + cited:**
```json
{ "text": "The RCM was paid via DRC-03 before filing to avoid a notice and penalty exposure … [ANN-2425-FY-0012 p3]",
  "intent": "llm", "tier": "cheap_llm", "llm_used": true, "model": "deepseek-chat",
  "cost_usd": 0.000217, "sources": [{"report_id":"ANN-2425-FY-0012","page":3}] }
```

**Out of scope (closed loop) — one line, muted:**
```json
{ "text": "Out of scope. I only cover the Meridian Rakshak reports (ITC, GSTR-9/9C, TDS, notice).",
  "intent": "out_of_scope", "tier": "refused", "in_scope": false, "cost_usd": 0.0 }
```

---

## 4. The 4 preloaded questions (chips)
Render as tappable chips above the input; also from `GET /suggested`:
1. What should I prioritise before the deadlines?
2. Which statutory windows are closing?
3. What should I do about Pinnacle Advisory?
4. What is the notice position?

---

## 4a2. Access gate (REQUIRED — the backend enforces it)

The assistant is locked behind an access code. The backend rejects `/ask`,
`/chat` and `/suggested` with **401** unless the request carries a valid code in
the **`X-Access-Key`** header. `/health` stays open.

- Show a **passcode screen** first: the user TYPES the access code (never hard-code
  it in the app / bundle — that would defeat the gate). Store it in
  `sessionStorage` (`rakshak.key`).
- Send it on EVERY request to the assistant:
  `headers: { "Content-Type":"application/json", "X-Access-Key": accessCode }`.
- On **401**, clear the stored code and return to the passcode screen with
  "invalid or expired access code."
- Do not display or log the code anywhere.

```js
const accessCode = sessionStorage.getItem("rakshak.key");
const res = await fetch(`${API_URL}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-Access-Key": accessCode },
  body: JSON.stringify({ session_id, messages }),
});
if (res.status === 401) { sessionStorage.removeItem("rakshak.key"); showPasscodeScreen(); }
```

## 4b. Conversation state & reliability (frontend owns the memory)

The API is **stateless** — `/chat` remembers nothing. The frontend keeps the
conversation and its durability. Implement ALL of the following:

- **State:** hold `messages: {role:"user"|"assistant", content:string}[]` AND a
  `session_id` (a UUID generated once when the conversation starts). On send,
  append the user msg, POST `{ session_id, messages }` (the WHOLE array), render
  `reply.text`, then append `{role:"assistant", content: reply.text}`. Send only
  `user`/`assistant` roles (the server adds the system prompt). **Always include
  `session_id`** — the server uses it to save the whole session for review. On
  "New chat", generate a NEW `session_id`.
- **Persist across refresh/reload/close:** write `messages[]` to `localStorage`
  (key `rakshak.chat.v1`) on every change and rehydrate on load. WITHOUT this a
  page refresh loses the conversation. Provide a **"New chat"** button that clears
  the array + the key. (No cross-device memory — there is no backend DB/login.)
- **Model memory is the last ~12 messages** (server sliding window). Store/display
  the full chat, but don't expect the model to recall very old turns.
- **Timeouts:** a `/chat` turn takes 10–30s (tool loop) and the host cold-starts
  ~50s after idle → client timeout **≥ 60–90s** + a "thinking… (first message may
  take ~1 min)" state.
- **No double-submit:** disable send while a request is in flight (each call costs
  money and re-runs the model).
- **Retry only on failure, never on success:** on error/timeout keep the user's
  message, do NOT append an assistant bubble, show an inline retry. A successful
  call already spent tokens — never auto-resend it.
- **Optimistic user bubble**, marked failed if the request errors.
- **Handle** `{error:"..."}` (400/404) and 5xx with a friendly retry; handle
  `reply.fell_back === true` (model unavailable → answered offline single-shot) with
  a subtle "offline" hint.
- **Cap** stored history (~last 50 messages) so localStorage stays small.
- **Privacy:** this is compliance data — choose `localStorage` (persists) vs
  `sessionStorage` (clears on tab close) deliberately, and offer a "clear" control.
- Optional: multi-tab sync via the `storage` event; a keep-warm ping to avoid cold
  starts.

## 5. Rendering rules (important for the "CA-grade" feel)
- Render `text` **verbatim**; convert `\n` to line breaks. Do **not** summarise.
- If a bubble's text has bullet lines (`•`) or labelled lines (`Verdict:`,
  `Amount:`, `Statutory basis:`, `Proof:`), render it in a **monospace** block —
  it's a ledger/decision card and alignment reads as authoritative.
- Show `sources[]` as small clickable chips: `ANN-2425-FY-0012 · p1` (Courier).
- Show a tiny provenance/cost footer per assistant bubble:
  - `tier` badge: deterministic → green "offline · $0"; cheap_llm/capable_llm →
    blue "AI · $" + `cost_usd.toFixed(6)` + " · " + `model`;
    response_cache/semantic_cache → grey "cached · $0"; refused → grey "out of scope".
- `in_scope === false` (refusal): render muted/italic, no source chips, no cost.
- Urgency words in advisories — style inline: `OVERDUE`/`critical` red,
  `soon` amber, `ok` muted green.

## 6. Behaviours
- On send: optimistic user bubble, typing indicator until `/ask` resolves.
- Network error: inline retry ("Couldn't reach the agent — retry").
- No message history persistence needed (each `/ask` is stateless).
- No login, no settings, no file upload, no external links. Closed loop only.

---

## 7. Visual system (match the Rakshak reports)
Professional, restrained, "compliance document" aesthetic.
```css
--ink:#1a1f2b; --muted:#6b7280; --line:#d8d5cc; --line-soft:#e7e4dc;
--cream:#faf8f3; --green:#1f6f43; --red:#a83232; --amber:#946200;
--blue:#274b74; --navy:#1e2a3a; --gold:#C9A227;
```
- Background `--cream`; text `--ink`; hairline dividers `--line-soft`.
- Header: brand **"Rakshak Systems"** in **Georgia serif**, small-caps subtitle
  "SentinelXOS · Virtual CA · deterministic".
- Body/UI: Helvetica/Arial. Ledger cards, IDs, hashes, gates: **Courier mono**.
- Assistant bubbles cream with a left 3px accent bar (green for offline/advisory,
  blue when `llm_used`). User bubbles subtle ink-tint.
- Amounts use Indian grouping as returned (₹1,80,000). Keep it tight and quiet —
  no gradients, no emoji, no marketing chrome.

Deliver a clean, single-screen chat that feels like a compliance instrument.
