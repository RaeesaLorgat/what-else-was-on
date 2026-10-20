"""Draw a quick picture of a generated match, to check it looks like football.

Usage:
    python scripts/preview_match.py --seed 1            # after generate_match.py
    python scripts/preview_match.py --seed 1 --minute 23

Saves data/match_<seed>_preview.png with two panels:
- left: everyone's position at one moment, the passer's vision cone, and the pass played;
- right: every pass by team 0 in the first 15 minutes (green = completed, red = lost).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge, Rectangle, Circle  # noqa: E402

from whatelse.generator import load_match, PITCH_X, PITCH_Y  # noqa: E402

NAVY, BLUE, GREY, GREEN, RED = "#1E2761", "#5A6AA8", "#6B7280", "#2C5F2D", "#B42318"


def pitch(ax):
    ax.add_patch(Rectangle((0, 0), PITCH_X, PITCH_Y, fill=False, color="#9AA0A6"))
    ax.plot([PITCH_X / 2] * 2, [0, PITCH_Y], color="#9AA0A6", lw=1)
    ax.add_patch(Circle((PITCH_X / 2, PITCH_Y / 2), 9.15, fill=False, color="#9AA0A6"))
    for x0 in (0, PITCH_X - 16.5):
        ax.add_patch(Rectangle((x0, PITCH_Y / 2 - 20.16), 16.5, 40.32, fill=False, color="#9AA0A6"))
    ax.set_xlim(-2, PITCH_X + 2)
    ax.set_ylim(-2, PITCH_Y + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--minute", type=float, default=None, help="show the first pass after this minute")
    p.add_argument("--data", default="data")
    a = p.parse_args()
    meta, trk = load_match(a.seed, a.data)
    ev, players, hz = meta["events"], meta["players"], meta["hz"]
    passes = [e for e in ev if e["type"] == "pass"]
    after = (a.minute or 0) * 60
    ps = next(e for e in passes if e["t"] >= after)
    f = ps["frame"]
    pos, ball = trk["players"][f], trk["ball"][f]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    ax = axes[0]
    pitch(ax)
    i = ps["player"]
    prev = trk["players"][max(f - 5, 0), i]
    vx, vy = pos[i][0] - prev[0], pos[i][1] - prev[1]
    if math.hypot(vx, vy) < 0.4:                # standing still: assume facing the goal he attacks
        vx, vy = (1, 0) if players[i]["team"] == 0 else (-1, 0)
    ang = math.degrees(math.atan2(vy, vx))
    ax.add_patch(Wedge(pos[i], 30, ang - 60, ang + 60, color=BLUE, alpha=0.12))
    for pl in players:
        x, y = pos[pl["pid"]]
        c = NAVY if pl["team"] == 0 else GREY
        ax.add_patch(Circle((x, y), 1.1, color=c))
        ax.text(x, y, str(pl["jersey"]), color="white", ha="center", va="center", fontsize=7)
    ax.annotate("", xy=(ps["end_x"], ps["end_y"]), xytext=(ps["x"], ps["y"]),
                arrowprops=dict(arrowstyle="->", color=GREEN if ps["outcome"] == "complete" else RED, lw=2))
    ax.add_patch(Circle(ball, 0.6, color="white", ec="black", zorder=5))
    who = players[i]
    m, s = divmod(int(ps["t"]), 60)
    ax.set_title(f"{m}:{s:02d}  #{who['jersey']} {who['name']} passes to #{players[ps['target']]['jersey']} "
                 f"({ps['outcome']}, {'in view' if ps['extra']['in_view'] else 'blind'})", fontsize=10)

    ax = axes[1]
    pitch(ax)
    for e in passes:
        if e["team"] != 0 or e["t"] > 15 * 60:
            continue
        ax.annotate("", xy=(e["end_x"], e["end_y"]), xytext=(e["x"], e["y"]),
                    arrowprops=dict(arrowstyle="->", color=GREEN if e["outcome"] == "complete" else RED,
                                    lw=0.8, alpha=0.6))
    t0 = meta["teams"][0]
    ax.set_title(f"{t0['name']} passes, first 15 minutes (attacking right): green = completed, red = lost",
                 fontsize=10)
    s = meta["summary"]
    fig.suptitle(f"Synthetic match, seed {a.seed}: {meta['teams'][0]['name']} {s['score'][0]} - "
                 f"{s['score'][1]} {meta['teams'][1]['name']}", fontsize=12)
    out = Path(a.data) / f"match_{a.seed}_preview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
