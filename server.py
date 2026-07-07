#!/usr/bin/env python3
"""
Minimal HTTP API for the Rakshak virtual-CA agent (stdlib only, no frameworks).

This is the backend a chatbot frontend calls. Endpoints:

  GET  /health           -> {"ok": true, "reports": 5, ...}
  GET  /suggested        -> {"questions": [ ...4 preloaded... ]}
  POST /ask  {"question": "..."}  -> SmartAnswer JSON (text, tier, cost_usd, ...)

Run:
    # offline only (zero cost):
    python server.py
    # with DeepSeek escalation for hard questions:
    set DEEPSEEK_API_KEY=sk-...        (Windows)  /  export on macOS/Linux
    python server.py --port 8000

CORS is open so a locally-served frontend can call it directly.
"""

import argparse
import datetime
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from rakshak_agent import (SmartAgent, ChatAgent, DeepSeekClient,
                           DeepSeekChatClient, suggested_questions)


class SessionStore:
    """Stores each chat SESSION (full transcript), keyed by session_id and
    upserted on every /chat turn. Because the frontend sends the whole history
    each call, the stored copy is always the complete conversation so far — so a
    finished session is just its final state. A puller on your PC mirrors these
    to local transcript files. Also prints to stdout (Render Logs), optionally
    appends to SESSION_FILE and forwards to SESSION_WEBHOOK_URL."""

    def __init__(self):
        self.sessions = {}                 # session_id -> record
        self._version = 0
        self._lock = threading.Lock()
        self.webhook = os.environ.get("SESSION_WEBHOOK_URL")
        self.file = os.environ.get("SESSION_FILE")

    def upsert(self, session_id, messages, meta=None):
        try:
            with self._lock:
                self._version += 1
                now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                rec = self.sessions.get(session_id) or {"session_id": session_id,
                                                        "created_ts": now}
                rec.update({"version": self._version, "updated_ts": now,
                            "n_messages": len(messages), "messages": messages,
                            "meta": meta or {}})
                self.sessions[session_id] = rec
                snap = dict(rec)
            print("SESSION v%d %s (%d msgs)" % (snap["version"], session_id,
                                                snap["n_messages"]), flush=True)
            line = json.dumps(snap, ensure_ascii=False)
            if self.file:
                try:
                    with open(self.file, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except Exception:
                    pass
            if self.webhook:
                threading.Thread(target=self._forward, args=(line,), daemon=True).start()
        except Exception:
            pass   # capture must never break a response

    def _forward(self, line):
        try:
            import urllib.request
            req = urllib.request.Request(
                self.webhook, data=line.encode("utf-8"),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass

    def since(self, version):
        with self._lock:
            return sorted((r for r in self.sessions.values() if r["version"] > version),
                          key=lambda r: r["version"])


SESSIONS = SessionStore()
LOG_TOKEN = os.environ.get("LOG_TOKEN")   # required to read GET /sessions


def build_agents():
    """SmartAgent (deterministic + cascade, powers /ask) and ChatAgent
    (conversational tool-using agent, powers /chat)."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    cheap = capable = chat_client = None
    if key:
        cheap = DeepSeekClient(name="deepseek-chat", tier="cheap")
        capable = DeepSeekClient(name="deepseek-reasoner", tier="capable",
                                 model="deepseek-reasoner",
                                 price_in_per_m=0.55, price_out_per_m=2.19)
        chat_client = DeepSeekChatClient(name="deepseek-chat", model="deepseek-chat")
    sa = SmartAgent.load("knowledge", cheap_llm=cheap, capable_llm=capable)
    ca = ChatAgent(sa.agent, chat_client=chat_client)
    return sa, ca


AGENT = None   # SmartAgent — set in main()
CHAT = None    # ChatAgent  — set in main()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path == "/sessions":
            return self._serve_sessions(parse_qs(parsed.query))
        if path in ("", "/", "/index.html"):
            self._send(200, {"name": "Rakshak virtual-CA agent API",
                             "endpoints": ["GET /health", "GET /suggested",
                                           "POST /ask {question}"]})
        elif path == "/health":
            s = AGENT.kb.stats()
            self._send(200, {"ok": True,
                             "llm_enabled": AGENT.router.cheap is not None,
                             "chat_enabled": CHAT is not None and CHAT.client is not None,
                             **s})
        elif path == "/suggested":
            self._send(200, {"questions": suggested_questions()})
        else:
            self._send(404, {"error": "not found"})

    def _serve_sessions(self, qs):
        token = (qs.get("token", [None])[0] or self.headers.get("X-Log-Token"))
        if not LOG_TOKEN:
            return self._send(403, {"error": "session read is disabled (set LOG_TOKEN)"})
        if token != LOG_TOKEN:
            return self._send(401, {"error": "bad or missing log token"})
        try:
            since = int(qs.get("since", ["0"])[0])
        except ValueError:
            since = 0
        recs = SESSIONS.since(since)
        last = recs[-1]["version"] if recs else since
        self._send(200, {"sessions": recs, "last_version": last, "count": len(recs)})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "invalid JSON body"})

        if path == "/ask":
            question = (data.get("question") or "").strip()
            context = data.get("context")
            if not question:
                return self._send(400, {"error": "question is required"})
            ans = AGENT.ask(question, context=context if isinstance(context, dict) else None)
            return self._send(200, ans.to_dict())

        if path == "/chat":
            # conversational agent: full history [{role, content}, ...]
            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._send(400, {"error": "messages array is required"})
            reply = CHAT.chat(messages)
            # Capture the whole session (transcript incl. this answer) keyed by a
            # stable session_id from the frontend. Latest state = finished session.
            session_id = data.get("session_id") or ("no-sid-%d" % SESSIONS._version)
            transcript = list(messages) + [{"role": "assistant", "content": reply.text}]
            SESSIONS.upsert(session_id, transcript,
                            {"model": reply.model, "cost_usd": reply.cost_usd,
                             "fell_back": reply.fell_back,
                             "tools_used": reply.tools_used})
            return self._send(200, reply.to_dict())

        self._send(404, {"error": "not found"})

    def log_message(self, *a):  # quiet
        pass


def main():
    global AGENT, CHAT
    ap = argparse.ArgumentParser()
    # Cloud hosts (Render, HF Spaces, Cloud Run, Fly...) inject $PORT and expect
    # the process to bind 0.0.0.0. Both are overridable for local dev.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args()

    AGENT, CHAT = build_agents()
    llm = "on (DeepSeek)" if AGENT.router.cheap else "off (offline only, $0)"
    print("Rakshak agent API on http://%s:%d  · LLM: %s"
          % (args.host, args.port, llm), flush=True)
    logs = "on (LOG_TOKEN set)" if LOG_TOKEN else "off (set LOG_TOKEN to enable)"
    print("  GET /  GET /health  GET /suggested", flush=True)
    print("  POST /ask  {question}                 (deterministic, $0 for most)", flush=True)
    print("  POST /chat {session_id, messages}     (conversational agent + tools)", flush=True)
    print("  GET  /sessions?since=&token=          (session capture: %s)" % logs, flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
