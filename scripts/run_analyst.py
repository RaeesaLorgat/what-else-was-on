"""Run the Analyst agent (Microsoft Agent Framework + your Foundry model) on passes.

Usage:
    python scripts/run_analyst.py --seed 1 --pass 1447     # one pass
    python scripts/run_analyst.py --seed 1 --top 3         # the 3 biggest missed options in the match

Needs .env filled in (same as smoke_test.py) and a match generated with
scripts/generate_match.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.agents import analyse_with_agent  # noqa: E402
from whatelse.tools import get_match, find_missed_passes  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pass", dest="pass_id", type=int, default=None)
    p.add_argument("--top", type=int, default=1)
    p.add_argument("--data", default="data")
    a = p.parse_args()
    m = get_match(a.seed, a.data)
    ids = [a.pass_id] if a.pass_id is not None else [r["pass_id"] for r in find_missed_passes(m, limit=a.top)]
    for pid in ids:
        print(f"--- Analyst on pass {pid} ---")
        out = await analyse_with_agent(m, pid)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
