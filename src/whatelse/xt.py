"""Expected threat (xT): how likely a move is to end in a goal soon if the ball is at a given spot.

Two maps are available:

- Karun Singh's public 12 x 8 xT grid, learned from a season of real Premier League
  event data (https://karun.in/blog/expected-threat.html). It is not stored in this
  repo: run `python scripts/get_xt_grid.py` once to download it into data/.
- A hand-made stand-in (generator.threat), used when the grid has not been downloaded.

The match generator always uses the hand-made map to decide where players pass. The
tools judge passes with Karun's grid when it is available, so the judge is independent
of the rule the simulated players follow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .generator import threat as handmade_threat, PITCH_X, PITCH_Y

XT_URL = "https://karun.in/blog/data/open_xt_12x8_v1.json"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "karun_xt_12x8.json"

_grid: np.ndarray | None = None
_loaded = False


def check_grid(raw) -> np.ndarray:
    """The file is 8 rows (across the pitch) of 12 columns (along it), attacking left to right.
    As in socceraction, the first row is the far touchline (y = 68), so we flip it."""
    g = np.asarray(raw, dtype=float)
    if g.shape != (8, 12) or not np.all((g >= 0) & (g <= 1)):
        raise ValueError(f"expected an 8 x 12 grid of probabilities, got shape {g.shape}")
    return g[::-1].copy()          # row 0 = y near 0


def set_grid(raw) -> None:
    """Use this grid (8 x 12 list, as in the file), or None for the hand-made map."""
    global _grid, _loaded
    _grid = None if raw is None else check_grid(raw)
    _loaded = True


def _active() -> np.ndarray | None:
    global _loaded
    if not _loaded:
        path = Path(os.environ.get("WHATELSE_XT", DEFAULT_PATH))
        if path.exists():
            set_grid(json.loads(path.read_text()))
        _loaded = True
    return _grid


def model_name() -> str:
    return "Karun Singh 12x8 xT grid" if _active() is not None else "hand-made xT-style map"


def grid_value(g: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation between cell centres, so the value does not jump at cell edges."""
    rows, cols = g.shape
    u = min(max(x / (PITCH_X / cols) - 0.5, 0.0), cols - 1.0)
    v = min(max(y / (PITCH_Y / rows) - 0.5, 0.0), rows - 1.0)
    c0, r0 = int(u), int(v)
    c1, r1 = min(c0 + 1, cols - 1), min(r0 + 1, rows - 1)
    fu, fv = u - c0, v - r0
    top = g[r0, c0] * (1 - fu) + g[r0, c1] * fu
    bot = g[r1, c0] * (1 - fu) + g[r1, c1] * fu
    return float(top * (1 - fv) + bot * fv)


def xt(x: float, y: float, attack_right: bool) -> float:
    g = _active()
    if g is None:
        return handmade_threat(x, y, attack_right)
    if not attack_right:                     # turn the pitch round so the team attacks left to right
        x, y = PITCH_X - x, PITCH_Y - y
    return grid_value(g, x, y)
