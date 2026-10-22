"""Fit the completion-probability (xP) curve used by the tools.

For every pass in the given synthetic matches, compute the arrival margin (how many
seconds the ball has to spare over the quickest defender) and whether the pass was
completed, then fit a one-variable logistic regression:

    xP = 1 / (1 + exp(-(a + b * margin)))

Usage:
    python scripts/fit_xp.py --seeds 1 2 3

Prints a and b (copy them into XP_INTERCEPT, XP_SLOPE in src/whatelse/tools.py),
the AUC, and a calibration table (predicted vs actual completion by margin).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.tools import get_match, arrival  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--data", default="data")
    a = p.parse_args()
    X, y = [], []
    for seed in a.seeds:
        m = get_match(seed, a.data)
        for e in m.passes:
            if e["outcome"] not in ("complete", "intercepted", "out"):
                continue
            r = arrival(m, e["player"], e["target"], e["frame"])
            X.append(float(np.clip(r["closest_call_s"], -2, 2)))
            y.append(e["outcome"] == "complete")
    X, y = np.array(X), np.array(y, float)
    A = np.c_[np.ones(len(X)), X]
    w = np.zeros(2)
    for _ in range(30000):
        pr = 1 / (1 + np.exp(-A @ w))
        w += 0.5 * A.T @ (y - pr) / len(y)
    pr = 1 / (1 + np.exp(-A @ w))
    order = np.argsort(pr)
    rank = np.empty(len(pr))
    rank[order] = np.arange(len(pr))
    pos = y == 1
    auc = (rank[pos].mean() - (pos.sum() - 1) / 2) / (~pos).sum()
    print(f"{len(y)} passes, {100 * y.mean():.1f}% completed")
    print(f"XP_INTERCEPT, XP_SLOPE = {w[0]:.3f}, {w[1]:.3f}   (AUC {auc:.3f})")
    print("margin (s)        passes  actual  predicted")
    for lo, hi in [(-2, -0.3), (-0.3, 0), (0, 0.3), (0.3, 0.6), (0.6, 2.01)]:
        s = (X >= lo) & (X < hi)
        if s.any():
            print(f"{lo:>5} to {hi:<5}   {s.sum():>8}  {100 * y[s].mean():>5.0f}%  {100 * pr[s].mean():>8.0f}%")


if __name__ == "__main__":
    main()
