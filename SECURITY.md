# Access gate (hard, backend-enforced)

The assistant endpoints (`/ask`, `/chat`, `/suggested`) require a valid access
code in the `X-Access-Key` header. The server enforces it; a client-side lock
alone would be bypassable, so this is the real gate. `/health` stays open (for
uptime/keep-warm). `/sessions` is separately protected by `LOG_TOKEN`.

## Turn it on (Render → Environment)
```
ACCESS_KEYS = code-for-ca-anita,code-for-ca-ravi,code-for-firm-desk
```
- Comma-separated → **one code per user** so you can revoke individually (delete
  a code, save → redeploys → that code stops working; others keep working).
- Or a single shared code: `ACCESS_KEY = one-shared-secret`.
- If neither is set, the gate is **OPEN** (dev only). `/health` shows
  `"access_gate": true/false`.

Optional hardening:
```
ALLOWED_ORIGIN = https://your-site.netlify.app   # lock CORS to your site (default *)
```

## How the frontend uses it
The passcode screen asks the user to TYPE the code (never bake it into the
build). It's stored in `sessionStorage` and sent as `X-Access-Key` on every
assistant request. On `401` the app clears it and re-prompts. See
`frontend/CLAUDE_DESIGN_BRIEF.md` §4a2.

## Rotate / revoke
- **Revoke one user:** remove their code from `ACCESS_KEYS`, save.
- **Rotate all:** change the codes, tell users the new one.
- Codes live only in Render env + the user's browser session — never in git.

## Honest limits & recommended backstop
- A shared/typed code controls WHO gets in, but a code holder can still send many
  requests. Because `/chat` calls DeepSeek, add a **cost backstop** so a leaked or
  over-used code can't blow up spend:
  - a **daily spend cap** (stop model calls past $X/day), and
  - a **per-IP rate limit**.
  (Not yet implemented — ask and I'll add both; ~60 lines, env-driven.)
- Comparison is constant-time (`hmac.compare_digest`); traffic is HTTPS on Render,
  so the header isn't sent in clear text.
