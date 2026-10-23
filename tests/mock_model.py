"""A tiny stand-in for the Foundry model, used only by the tests.

It speaks the OpenAI chat-completions format on localhost and plays all the agents,
telling them apart by their instructions:

- Analyst:  asks for analyse_pass, then writes the verdict from the tool numbers.
            With sloppy=True its FIRST draft has a wrong gain, to test that the
            Checker catches it and the Analyst fixes it.
- Checker:  asks for verify_verdict, then approves only if it found no problems.
- Narrator / Recap: write copy from the verdict numbers.

This checks the wiring (tools, messages, JSON, the workflow loop) without calling Azure.
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


def _text(m) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


def _first_json(text: str):
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def _call(name: str, args: dict) -> dict:
    return _reply(tool_calls=[{"id": "call_1", "type": "function",
                               "function": {"name": name, "arguments": json.dumps(args)}}])


def _tool_result(msgs) -> dict:
    a = json.loads(_text([m for m in msgs if m.get("role") == "tool"][-1]))
    return json.loads(a) if isinstance(a, str) else a


class _Handler(BaseHTTPRequestHandler):
    requests: list = []
    sloppy: bool = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append(body)
        msgs = body["messages"]
        system = " ".join(_text(m) for m in msgs if m.get("role") in ("system", "developer"))
        user = [_text(m) for m in msgs if m.get("role") == "user"][-1]
        has_tool = any(m.get("role") == "tool" for m in msgs)
        if "checking broadcast copy" in system:
            out = _reply(content=json.dumps({"approved": True, "problems": []}))
        elif "You are the Checker" in system:
            if not has_tool:
                out = _call("verify_verdict", {"verdict_json": json.dumps(_first_json(user))})
            else:
                r = _tool_result(msgs)
                out = _reply(content=json.dumps({"approved": not r["problems"], "problems": r["problems"]}))
        elif "post-match recap" in system:
            vs = json.JSONDecoder().raw_decode(user[user.index("["):])[0]
            out = _reply(content=json.dumps({"recap": " ".join(
                f"At {v['time']}, {v['passer']} could have found {v['better_option']}." for v in vs)}))
        elif "You are the Narrator" in system:
            v = _first_json(user)
            out = _reply(content=json.dumps({
                "commentary": f"{v['passer']} plays it to {v['pass_played_to']}.",
                "pundit": f"{v['better_option']} was on, a {v['evidence']['better_option_xp_pct']}% chance that pass comes off."
                           + (" It would have been tight." if (v["evidence"]["closest_call_s"] or 1) < 0.3 else "")}))
        elif not has_tool:
            pid = int(re.search(r"pass (\d+)", user).group(1))
            out = _call("analyse_pass", {"pass_id": pid})
        else:
            a = _tool_result(msgs)
            best = next((o for o in a["options"] if o["better_option"]), None)
            first_draft = "rejected" not in user
            gain = best["gain_vs_pass_played_pct"] if best else None
            if best and _Handler.sloppy and first_draft:
                gain = round(gain + 1.0, 1)          # a deliberate mistake for the Checker to catch
            verdict = {
                "pass_id": a["pass_id"], "time": a["time"],
                "verdict": "missed_better_option" if best else "good_choice",
                "passer": a["passer"], "pass_played_to": a["pass_played_to"],
                "better_option": best["label"] if best else None,
                "evidence": {"threat_of_pass_played_pct": a["threat_of_pass_played_pct"],
                             "better_option_threat_pct": best["threat_pct"] if best else None,
                             "gain_pct": gain,
                             "better_option_in_view": best["in_view"] if best else None,
                             "better_option_offside": best["offside"] if best else None,
                             "better_option_arrives": best["arrives"] if best else None,
                             "better_option_xp_pct": best["xp_pct"] if best else None,
                             "closest_call_s": best["closest_call_s"] if best else None},
                "explanation": "Mock explanation." + (" It would have been tight."
                                                      if best and best["closest_call_s"] < 0.3 else ""),
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
