"""Connection check: can we reach the model in Foundry, and can it call a tool?

Usage:
    python scripts/smoke_test.py          # real call to your Foundry deployment
    python scripts/smoke_test.py --mock   # no Azure needed; checks the code path

Check 1 asks the model for a one-line reply.
Check 2 gives the model a small football tool (distance between two players)
and checks it calls the tool and uses the answer. The agents need this later.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.llm import load_settings, make_client  # noqa: E402


def distance_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """Distance in metres between two points on a 105 x 68 m pitch."""
    return round(math.hypot(x2 - x1, y2 - y1), 2)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "distance_m",
        "description": "Distance in metres between two points on a 105 x 68 m pitch.",
        "parameters": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in ("x1", "y1", "x2", "y2")},
            "required": ["x1", "y1", "x2", "y2"],
        },
    },
}]

QUESTION = ("The passer is at (40, 30) and a teammate is at (58, 22). "
            "Use the tool to get the distance, then answer in one sentence.")


class MockClient:
    """Stands in for the Foundry client so the script runs without Azure."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, tools=None, **_):
        last = messages[-1]
        if tools and not any(m.get("role") == "tool" for m in messages if isinstance(m, dict)):
            call = SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(
                name="distance_m", arguments=json.dumps({"x1": 40, "y1": 30, "x2": 58, "y2": 22})))
            msg = SimpleNamespace(content=None, tool_calls=[call], role="assistant")
        elif tools:
            msg = SimpleNamespace(content=f"The teammate is {last['content']} m away.", tool_calls=None, role="assistant")
        else:
            msg = SimpleNamespace(content="OK, connected.", tool_calls=None, role="assistant")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def check_reply(client, model) -> None:
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Reply with exactly: OK, connected."}])
    print("Check 1, model reply:", r.choices[0].message.content)


def check_tool(client, model) -> None:
    messages = [{"role": "user", "content": QUESTION}]
    r = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    msg = r.choices[0].message
    if not msg.tool_calls:
        raise SystemExit("Check 2 FAILED: the model answered without calling the tool:\n" + str(msg.content))
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    result = distance_m(**args)
    print(f"Check 2, tool called: distance_m({args}) -> {result}")
    messages.append({"role": "assistant", "content": None, "tool_calls": [{
        "id": call.id, "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments}}]})
    messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    r2 = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    print("Check 2, final answer:", r2.choices[0].message.content)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="run without Azure")
    a = p.parse_args()
    if a.mock:
        client, model = MockClient(), "mock-model"
        print("Running in mock mode (no Azure calls).")
    else:
        s = load_settings()
        client, model = make_client(s), s.model
        print(f"Calling deployment '{s.model}' at {s.base_url}")
    check_reply(client, model)
    check_tool(client, model)
    print("All checks passed.")


if __name__ == "__main__":
    main()
