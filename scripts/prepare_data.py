"""Make sure the data the app needs is in data/ (used when building the container).

- matches 1, 2 and 3: generated if missing (about a minute each)
- Karun Singh's xT grid: downloaded if missing (the tools fall back to a hand-made map if that fails)

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("WHATELSE_DATA", ROOT / "data"))
SEEDS = (1, 2, 3)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for seed in SEEDS:
        if (DATA / f"match_{seed}_events.json").exists() and (DATA / f"match_{seed}_tracking.npz").exists():
            print(f"match {seed}: already there")
            continue
        print(f"match {seed}: generating...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_match.py"), "--seed", str(seed),
                        "--out", str(DATA)], check=True)
    if (DATA / "karun_xt_12x8.json").exists():
        print("xT grid: already there")
    else:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "get_xt_grid.py")])
        if r.returncode != 0:
            print("xT grid: download failed; the tools will use the hand-made map")
    stories = sorted(p.name for p in DATA.glob("stories_*.json"))
    print("stories:", ", ".join(stories) if stories else "none yet (run scripts/run_pipeline.py, or use the page)")


if __name__ == "__main__":
    main()
