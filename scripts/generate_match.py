"""Generate a synthetic match and save it to data/.

Usage:
    python scripts/generate_match.py                 # seed 1, full 90 minutes
    python scripts/generate_match.py --seed 7
    python scripts/generate_match.py --minutes 10    # quick test match

Writes data/match_<seed>_tracking.npz (positions at 10 frames per second)
and data/match_<seed>_events.json (teams, players, events, summary).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.generator import MatchSim  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--minutes", type=float, default=90)
    p.add_argument("--out", default="data")
    a = p.parse_args()
    t0 = time.time()
    print(f"Simulating {a.minutes:g} minutes with seed {a.seed}...")
    sim = MatchSim(seed=a.seed, minutes=a.minutes).run()
    path = sim.save(a.out)
    s = sim.summary()
    print(f"Done in {time.time() - t0:.0f} s. Saved {path} and its tracking file.")
    print(f"Score {s['score'][0]}-{s['score'][1]} | passes {s['passes']} ({100 * s['pass_completion']:.0f}% complete) | "
          f"blind passes {s['blind_passes']} | shots {s['shots']} | tackles {s['tackles']} | "
          f"interceptions {s['interceptions']} | team 0 possession {100 * s['possession_team0']:.0f}%")


if __name__ == "__main__":
    main()
