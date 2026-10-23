"""Run the full team on passes: Analyst -> Checker (-> Analyst again if rejected) -> Narrator.

Usage:
    python scripts/run_pipeline.py --seed 1 --pass 797           # one pass
    python scripts/run_pipeline.py --seed 1 --top 3 --recap      # the 3 biggest missed options, plus a recap
    python scripts/run_pipeline.py --seed 1 --top 3 --mock       # no Azure: a local stand-in model (for testing)

Prints what each agent did and saves everything to data/stories_<seed>.json (used by the replay page).
Needs .env filled in and a generated match.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.pipeline import run_pass, write_recap  # noqa: E402
from whatelse.tools import get_match, find_missed_passes  # noqa: E402


def show(story: dict) -> None:
    v = story["verdict"]
    print(f"\n=== pass {story['pass_id']} ({v.get('time')}) ===")
    for t in story["trail"]:
        a = t["agent"]
        if a == "Analyst":
            d = t["verdict"]
            what = (f"{d.get('verdict')} -> {d.get('better_option')} (+{(d.get('evidence') or {}).get('gain_pct')} pts)"
                    if d.get("verdict") == "missed_better_option" else d.get("verdict", "no JSON verdict"))
            print(f"  Analyst   draft {t['draft']}: {what}")
        elif a == "Checker":
            label = f"draft {t['draft']}" if "draft" in t else f"on {t.get('on')} copy"
            print(f"  Checker   {label}: {'approved' if t['approved'] else 'REJECTED'}")
            for p in t["problems"]:
                print(f"              - {p}")
        elif a == "Narrator":
            print(f"  Narrator  done")
        else:
            print(f"  {a:<9} {t.get('note')}")
    print(f"\n  Status:     {story['status']} after {story['drafts']} draft(s)")
    print(f"  Analyst:    {v.get('explanation')}")
    print(f"  Commentary: {story['commentary']}")
    print(f"  Pundit:     {story['pundit']}")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pass", dest="pass_id", type=int, default=None)
    p.add_argument("--top", type=int, default=1)
    p.add_argument("--recap", action="store_true")
    p.add_argument("--data", default="data")
    p.add_argument("--mock", action="store_true", help="use the local stand-in model from tests/ instead of Foundry")
    a = p.parse_args()
    client = None
    if a.mock:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        import mock_model
        from agent_framework.openai import OpenAIChatCompletionClient
        _, url = mock_model.start()
        client = OpenAIChatCompletionClient(model="mock", api_key="mock", base_url=url)
    m = get_match(a.seed, a.data)
    ids = [a.pass_id] if a.pass_id is not None else [r["pass_id"] for r in find_missed_passes(m, limit=a.top)]

    stories = []
    for pid in ids:
        s = await run_pass(m, pid, client=client)
        show(s)
        stories.append(s)

    out = {"seed": a.seed, "stories": stories}
    if a.recap:
        r = await write_recap(stories, client=client)
        out["recap"], out["recap_trail"] = r["recap"], r["trail"]
        print("\n=== Post-match recap ===")
        for t in r["trail"]:
            if t["agent"] == "Checker":
                print(f"  Checker   on recap: {'approved' if t['approved'] else 'REJECTED'}")
                for p_ in t["problems"]:
                    print(f"              - {p_}")
        print(out["recap"])

    path = Path(a.data) / f"stories_{a.seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    asyncio.run(main())
