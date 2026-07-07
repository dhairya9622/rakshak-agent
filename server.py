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
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rakshak_agent import SmartAgent, DeepSeekClient, suggested_questions


def build_agent():
    key = os.environ.get("DEEPSEEK_API_KEY")
    cheap = capable = None
    if key:
        cheap = DeepSeekClient(name="deepseek-chat", tier="cheap")
        capable = DeepSeekClient(name="deepseek-reasoner", tier="capable",
                                 model="deepseek-reasoner",
                                 price_in_per_m=0.55, price_out_per_m=2.19)
    return SmartAgent.load("knowledge", cheap_llm=cheap, capable_llm=capable)


AGENT = None  # set in main()


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
        path = self.path.rstrip("/")
        if path in ("", "/", "/index.html"):
            self._send(200, {"name": "Rakshak virtual-CA agent API",
                             "endpoints": ["GET /health", "GET /suggested",
                                           "POST /ask {question}"]})
        elif path == "/health":
            s = AGENT.kb.stats()
            self._send(200, {"ok": True, "llm_enabled": AGENT.router.cheap is not None, **s})
        elif path == "/suggested":
            self._send(200, {"questions": suggested_questions()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/ask":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            question = (data.get("question") or "").strip()
            context = data.get("context")  # optional: {"last_entity": "..."}
        except Exception:
            return self._send(400, {"error": "invalid JSON body"})
        if not question:
            return self._send(400, {"error": "question is required"})
        ans = AGENT.ask(question, context=context if isinstance(context, dict) else None)
        self._send(200, ans.to_dict())

    def log_message(self, *a):  # quiet
        pass


def main():
    global AGENT
    ap = argparse.ArgumentParser()
    # Cloud hosts (Render, HF Spaces, Cloud Run, Fly...) inject $PORT and expect
    # the process to bind 0.0.0.0. Both are overridable for local dev.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args()

    AGENT = build_agent()
    llm = "on (DeepSeek)" if AGENT.router.cheap else "off (offline only, $0)"
    print("Rakshak agent API on http://%s:%d  · LLM escalation: %s"
          % (args.host, args.port, llm), flush=True)
    print("  GET /  GET /health  GET /suggested  POST /ask {\"question\": \"...\"}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
