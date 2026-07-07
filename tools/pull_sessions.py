#!/usr/bin/env python3
"""
Bring every chat SESSION to THIS machine (your always-on PC).

Polls the server's token-protected /sessions endpoint and writes one readable
transcript per session to a local folder, updated as each session grows (so a
finished session is simply its final saved state). Remembers a cursor so it only
fetches what's new. Because it polls regularly it also keeps the free host warm
(no cold starts).

    python tools/pull_sessions.py \
        --url https://rakshak-agent.onrender.com \
        --token YOUR_LOG_TOKEN \
        --dir sessions \
        --interval 60

Set the SAME token as the server's LOG_TOKEN env var. Run it once (loops) or as
a scheduled task. Stdlib only. Outputs, per session:
    sessions/<updated_ts>_<session_id>.md   (human-readable transcript)
    sessions/_index.jsonl                    (one JSON line per pull, raw)
"""

import argparse
import json
import os
import re
import time
import urllib.request

ROLE_LABEL = {"user": "CA", "assistant": "Rakshak"}


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))[:80]


def load_cursor(d):
    try:
        with open(os.path.join(d, "_cursor")) as fh:
            return int(fh.read().strip() or "0")
    except Exception:
        return 0


def save_cursor(d, v):
    with open(os.path.join(d, "_cursor"), "w") as fh:
        fh.write(str(v))


def fetch(url, token, since):
    req = urllib.request.Request(
        "%s/sessions?since=%d" % (url.rstrip("/"), since),
        headers={"X-Log-Token": token})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def write_transcript(d, rec):
    sid = rec.get("session_id", "unknown")
    ts = rec.get("updated_ts", "")
    meta = rec.get("meta", {})
    fname = "%s_%s.md" % (_safe(ts).replace(":", ""), _safe(sid))
    # one stable file per session: remove older-timestamp files for this sid
    for old in os.listdir(d):
        if old.endswith("_%s.md" % _safe(sid)) and old != fname:
            try:
                os.remove(os.path.join(d, old))
            except Exception:
                pass
    lines = ["# Session %s" % sid,
             "%s → %s · %d messages · model=%s · est. cost $%s"
             % (rec.get("created_ts", ""), ts, rec.get("n_messages", 0),
                meta.get("model"), meta.get("cost_usd")),
             ""]
    for m in rec.get("messages", []):
        who = ROLE_LABEL.get(m.get("role"), m.get("role", "?"))
        lines.append("**%s:** %s" % (who, (m.get("content") or "").strip()))
        lines.append("")
    with open(os.path.join(d, fname), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def run_once(url, token, d):
    since = load_cursor(d)
    data = fetch(url, token, since)
    recs = data.get("sessions", [])
    if recs:
        with open(os.path.join(d, "_index.jsonl"), "a", encoding="utf-8") as idx:
            for rec in recs:
                write_transcript(d, rec)
                idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
        save_cursor(d, data["last_version"])
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="server base URL")
    ap.add_argument("--token", required=True, help="matches server LOG_TOKEN")
    ap.add_argument("--dir", default="sessions", help="local folder for transcripts")
    ap.add_argument("--interval", type=int, default=60, help="poll seconds (0 = once)")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    print("Mirroring chat sessions -> %s" % os.path.abspath(args.dir))
    while True:
        try:
            n = run_once(args.url, args.token, args.dir)
            if n:
                print("updated %d session(s), cursor=%d" % (n, load_cursor(args.dir)))
        except Exception as exc:
            print("poll error:", exc)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
