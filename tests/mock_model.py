"""A tiny stand-in for the Foundry model, used only by the tests.

It speaks the OpenAI chat-completions format on localhost: on the first turn it
asks for the analyse_pass tool; once it gets the tool result back, it writes the
JSON verdict from those numbers. This checks that the agent wiring (tools,
messages, JSON parsing) works without calling Azure.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def _reply(content=None, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "mock", "object": "chat.completion", "created": 0, "model": "mock",
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append(body)
        msgs = body["messages"]
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        if not tool_msgs:
            user = next(m for m in msgs if m["role"] == "user")
            text = user["content"] if isinstance(user["content"], str) else json.dumps(user["content"])
            pid = int(re.search(r"pass (\d+)", text).group(1))
            out = _reply(tool_calls=[{"id": "call_1", "type": "function",
                                      "function": {"name": "analyse_pass", "arguments": json.dumps({"pass_id": pid})}}])
        else:
            c = tool_msgs[-1]["content"]
            c = c if isinstance(c, str) else c[0]["text"]
            a = json.loads(c)
            if isinstance(a, str):
                a = json.loads(a)
            best = next((o for o in a["options"] if o["better_option"]), None)
            verdict = {
                "pass_id": a["pass_id"], "time": a["time"],
                "verdict": "missed_better_option" if best else "good_choice",
                "passer": a["passer"], "pass_played_to": a["pass_played_to"],
                "better_option": best["label"] if best else None,
                "evidence": {"threat_of_pass_played_pct": a["threat_of_pass_played_pct"],
                             "better_option_threat_pct": best["threat_pct"] if best else None,
                             "gain_pct": best["gain_vs_pass_played_pct"] if best else None,
                             "better_option_in_view": best["in_view"] if best else None,
                             "better_option_arrives": best["arrives"] if best else None,
                             "closest_call_s": best["closest_call_s"] if best else None},
                "explanation": "Mock explanation.",
            }
            out = _reply(content="```json\n" + json.dumps(verdict) + "\n```")
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/openai/v1/"
