# Deploy the agent for free, call it from a Netlify frontend

The agent runs as-is (Python, stdlib only) as a small web API. Host it free on
one of the options below, then point your Netlify frontend at its URL. The
DeepSeek key lives ONLY in the host's env vars — never in the frontend.

Runtime deps: **none** (stdlib). `knowledge/` is pre-generated and committed, so
nothing is built at request time. `pdfplumber` is only for the one-time
preprocessing (`requirements-preprocess.txt`), not for hosting.

First: push this folder to a GitHub repo (Render/HF read from GitHub).

---

## Option A — Render (recommended, simplest)

1. Push repo to GitHub.
2. render.com → **New +** → **Blueprint** → pick this repo. It reads `render.yaml`.
   (Or **New + → Web Service**, Build: `pip install -r requirements.txt`,
   Start: `python server.py`.)
3. In the service → **Environment** → add `DEEPSEEK_API_KEY = sk-...`
   (omit it to run offline-only at $0).
4. Deploy. Your URL is `https://rakshak-agent.onrender.com` (or similar).
5. Verify: open `<url>/health` → `{"ok":true,"llm_enabled":true,...}`.

Free tier note: the service **sleeps after ~15 min idle** (first request then
takes ~50s to wake). For a snappy chat, either keep it warm with a free pinger
(e.g. cron-job.org hitting `<url>/health` every 10 min) or use Option B.

## Option B — Hugging Face Spaces (free, no credit card)

1. huggingface.co → **New Space** → SDK: **Docker** → **Blank**.
2. Push this folder to the Space repo (it has a `Dockerfile`).
3. Space → **Settings → Variables and secrets** → add secret
   `DEEPSEEK_API_KEY`. Set **App port** to `8000`.
4. It builds and serves at `https://<user>-<space>.hf.space`.
   Verify `<url>/health`.

## Option C — any Docker host (Fly.io, Cloud Run, Koyeb, Railway)

```bash
docker build -t rakshak-agent .
docker run -p 8000:8000 -e DEEPSEEK_API_KEY=sk-... rakshak-agent
# then deploy the image to your chosen host; set DEEPSEEK_API_KEY there.
```

---

## Wire the Netlify frontend to it (cross-origin)

The API already sends open CORS headers, so the browser can call it directly.

1. In your Netlify site settings → **Environment variables**, add the API base
   URL, e.g. `NEXT_PUBLIC_API_URL = https://rakshak-agent.onrender.com`
   (Next.js) or `VITE_API_URL = ...` (Vite). Use that var in the fetch calls.
2. The generated UI calls:
   - `GET  ${API_URL}/suggested` on load (the 4 chips)
   - `POST ${API_URL}/ask  { "question": "..." }` per message
3. Deploy Netlify. Done — the frontend is static/free, the agent is hosted free,
   and the DeepSeek key never leaves the backend.

---

## Security
- Set `DEEPSEEK_API_KEY` only as a host env var/secret. It is never committed and
  never shipped to the browser (the frontend talks to your API, not DeepSeek).
- **Rotate the key you pasted in chat** — treat it as exposed.
- Optional hardening: replace `Access-Control-Allow-Origin: *` in `server.py`
  with your exact Netlify domain, and add a simple rate limit, before going public.
