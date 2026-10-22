"""Download Karun Singh's public 12 x 8 expected threat (xT) grid into data/.

The grid is Karun Singh's work (https://karun.in/blog/expected-threat.html), learned
from a season of Premier League event data. It is not stored in this repo; this script
fetches it from his site. Without it, the tools fall back to a hand-made map.

Usage:
    python scripts/get_xt_grid.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.xt import XT_URL, DEFAULT_PATH, check_grid  # noqa: E402


def main() -> None:
    with urllib.request.urlopen(XT_URL, timeout=30) as r:
        raw = json.loads(r.read().decode())
    g = check_grid(raw)                       # stops here if the file is not an 8 x 12 grid
    DEFAULT_PATH.parent.mkdir(exist_ok=True)
    DEFAULT_PATH.write_text(json.dumps(raw))
    print(f"Saved {DEFAULT_PATH.relative_to(Path.cwd()) if DEFAULT_PATH.is_relative_to(Path.cwd()) else DEFAULT_PATH}")
    print(f"8 x 12 grid, values from {100 * g.min():.1f}% to {100 * g.max():.1f}% (Karun Singh, karun.in)")


if __name__ == "__main__":
    main()
