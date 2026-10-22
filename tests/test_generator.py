import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.generator import MatchSim, load_match, PITCH_X, PITCH_Y, HZ  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    # 8 minutes keeps the tests quick (about 4 s)
    return MatchSim(seed=3, minutes=8).run()


def test_same_seed_same_match():
    a = MatchSim(seed=11, minutes=1).run()
    b = MatchSim(seed=11, minutes=1).run()
    assert np.array_equal(a.tracks, b.tracks)
    assert [e.type for e in a.events] == [e.type for e in b.events]


def test_shapes_and_bounds(sim):
    assert sim.tracks.shape == (8 * 60 * HZ, 22, 2)
    assert sim.tracks[..., 0].min() >= 0 and sim.tracks[..., 0].max() <= PITCH_X
    assert sim.tracks[..., 1].min() >= 0 and sim.tracks[..., 1].max() <= PITCH_Y


def test_players_move_at_human_speeds(sim):
    v = np.diff(sim.tracks, axis=0) * HZ
    speed = np.hypot(v[..., 0], v[..., 1])
    # kick-offs reset positions instantly, so allow a handful of jumps; everything else under 10 m/s
    assert (speed > 10).sum() <= 2 * 22
    assert np.percentile(speed, 99) < 8.5


def test_events_look_like_football(sim):
    c = Counter(e.type for e in sim.events)
    assert c["pass"] > 50
    passes = [e for e in sim.events if e.type == "pass"]
    assert all(e.outcome in ("complete", "intercepted", "out", "offside", "stopped") for e in passes[:-1])
    completion = sum(e.outcome == "complete" for e in passes) / len(passes)
    assert 0.65 < completion < 0.95
    # passes go from a player to a teammate
    for e in passes:
        assert sim.players[e.target].team == sim.players[e.player].team


def test_save_and_load(tmp_path, sim):
    sim.save(tmp_path)
    meta, trk = load_match(3, tmp_path)
    assert meta["synthetic"] is True
    assert len(meta["players"]) == 22
    assert trk["players"].shape == sim.tracks.shape
    assert meta["summary"]["passes"] == sum(e["type"] == "pass" for e in meta["events"])
