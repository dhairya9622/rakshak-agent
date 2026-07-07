# See every chat session on your PC

The server captures each chat **session** (the full transcript, keyed by
`session_id`, updated on every turn). A small puller on your always-on PC brings
them down as readable transcript files. The puller also polls regularly, which
keeps the free host warm (no cold starts).

## 1. One-time: turn capture on (Render)
In your Render service → **Environment** → add:
```
LOG_TOKEN = <make-up-a-long-secret>      # required to read /sessions
```
(Optional durability if you also want a server-side copy or a live feed:
`SESSION_FILE=/tmp/sessions.jsonl`, `SESSION_WEBHOOK_URL=<your webhook>`.)
Saving redeploys automatically.

## 2. Frontend must send a session_id
The chat UI generates a `session_id` (UUID) per conversation and includes it in
every `POST /chat` body: `{ session_id, messages }`. New "New chat" → new id.
(Already specified in `frontend/CLAUDE_DESIGN_BRIEF.md`.)

## 3. On your PC: run the puller
```
cd C:\Users\DHAIRYA\rakshak-agent
python tools\pull_sessions.py ^
    --url https://rakshak-agent.onrender.com ^
    --token <same LOG_TOKEN> ^
    --dir C:\Users\DHAIRYA\rakshak-sessions ^
    --interval 60
```
It writes, per session:
```
rakshak-sessions\<timestamp>_<session_id>.md   # readable transcript (CA: / Rakshak:)
rakshak-sessions\_index.jsonl                   # raw records (for search/analytics)
rakshak-sessions\_cursor                         # resume point
```
Each transcript is rewritten as the session grows, so a **finished session** is
simply its final saved file. Leave the puller running (it also prevents cold
starts). To run once instead of looping, use `--interval 0`.

### Keep it running (Windows Task Scheduler)
Create a Basic Task → Trigger: *At log on* / *Daily, repeat every 5 min* →
Action: Start a program → `python` with the arguments above. Or just leave a
terminal open on the always-on PC.

## Security & privacy
- `/sessions` is unreadable without `LOG_TOKEN`; keep it secret (it's only in
  Render env + your puller command, never in git).
- Transcripts are compliance conversations — store the local folder somewhere
  appropriate and back it up if needed.
- Capture is best-effort in memory on the free tier: the puller polling every
  ~60s keeps the server warm so sessions persist between polls; a manual redeploy
  clears the in-memory store, so anything not yet pulled at that moment is lost.
  Set `SESSION_WEBHOOK_URL` (e.g. to a Google-Sheet Apps Script or a logging
  service) if you need redeploy-proof durability.
