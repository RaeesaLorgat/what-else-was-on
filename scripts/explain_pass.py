"""Print the 'what else was on' analysis for passes in a generated match (no AI, just the tools).

Usage:
    python scripts/explain_pass.py --seed 1                 # list the 10 biggest missed options
    python scripts/explain_pass.py --seed 1 --pass 797      # full analysis of one pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.tools import get_match, analyse_pass, find_missed_passes  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pass", dest="pass_id", type=int, default=None)
    p.add_argument("--data", default="data")
    a = p.parse_args()
    m = get_match(a.seed, a.data)

    if a.pass_id is None:
        print(f"Biggest missed options in match {a.seed} (pass id, time, who, what was on):")
        for r in find_missed_passes(m, limit=10):
            print(f"  pass {r['pass_id']:>5}  {r['time']:>6}  {r['passer']} -> {r['pass_played_to']}, "
                  f"but {r['better_option']} was on (+{r['gain_pct']} pts threat, {r['xp_pct']}% to complete)")
        print("\nRun again with --pass <id> to see the full analysis of one pass.")
        return

    r = analyse_pass(m, a.pass_id)
    print(f"{r['time']}  {r['team']}: {r['passer']} passes to {r['pass_played_to']} "
          f"({r['pass_outcome']}, {'in view' if r['pass_played_in_view'] else 'blind'}); "
          f"threat where it went: {r['threat_of_pass_played_pct']}%")
    print(f"Something better on? {'YES: ' + r['best_better_option'] if r['missed_better_option'] else 'no'}\n")
    print(f"{'teammate':<22}{'in view':>8}{'offside':>8}{'dist m':>8}{'threat %':>10}{'vs played':>11}"
          f"{'arrives':>9}{'xP %':>6}  note")
    for o in r["options"]:
        note = ("<- pass played" if o["was_pass_played"] else
                "<- BETTER OPTION" if o["better_option"] else
                "offside" if o["offside"] and o["in_view"] and o["gain_vs_pass_played_pct"] >= 0.5 else
                f"cut out by {o['cut_out_by']}" if o["cut_out_by"] else
                "close call" if o["close_call"] else "")
        print(f"{o['label']:<22}{'yes' if o['in_view'] else 'no':>8}{'yes' if o['offside'] else 'no':>8}"
              f"{o['distance_m']:>8}{o['threat_pct']:>10}{o['gain_vs_pass_played_pct']:>+11}"
              f"{'yes' if o['arrives'] else 'no':>9}{o['xp_pct']:>6}  {note}")
    print(f"\nRule: {r['rule']}.")
    print(f"Threat model: {r['threat_model']}.")


if __name__ == "__main__":
    main()
