import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse.generator import MatchSim, load_match  # noqa: E402
from whatelse import tools  # noqa: E402
from whatelse.tools import Match, analyse_pass, find_missed_passes, vision_cone, pass_arrival  # noqa: E402


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    folder = tmp_path_factory.mktemp("m")
    MatchSim(seed=5, minutes=6).run().save(folder)
    return Match(*load_match(5, folder))


def test_cone_splits_teammates(match):
    p = match.passes[10]
    c = vision_cone(match, p["id"])
    assert len(c["teammates"]) == 10
    assert all((t["angle_from_facing_deg"] <= 60) == t["in_view"] for t in c["teammates"])


def test_arrival_worked_example():
    # one passer, one teammate 20 m straight ahead, one defender standing right on the line at 10 m:
    # the ball needs 0.8 s to get there; the defender needs only the 0.7 s reaction, so the pass is cut out
    meta = {"players": [{"pid": 0, "team": 0, "jersey": 1, "name": "A"}, {"pid": 1, "team": 0, "jersey": 2, "name": "B"},
                        {"pid": 2, "team": 1, "jersey": 3, "name": "C"}],
            "teams": [{"attacks": "right", "name": "X"}, {"attacks": "left", "name": "Y"}], "events": []}
    frames = np.zeros((5, 3, 2))
    frames[:, 0] = [40, 30]
    frames[:, 1] = [60, 30]
    frames[:, 2] = [50, 30]
    m = Match(meta, {"players": frames, "ball": np.zeros((5, 2))})
    r = tools.arrival(m, 0, 1, 4)
    assert r["arrives"] is False and r["cut_out_by"] == "#3 C"
    # move the defender 15 m off the line: now he cannot get there in time
    frames[:, 2] = [50, 45]
    r = tools.arrival(Match(meta, {"players": frames, "ball": np.zeros((5, 2))}), 0, 1, 4)
    assert r["arrives"] is True and r["closest_call_s"] > 0


def test_analyse_pass_is_consistent(match):
    found = 0
    for p in match.passes[:150]:
        a = analyse_pass(match, p["id"])
        json.dumps(a)                                  # the agent gets this as JSON
        for o in a["options"]:
            if o["better_option"]:
                assert o["in_view"] and o["arrives"] and not o["offside"] and o["gain_vs_pass_played_pct"] >= 100 * tools.BETTER_GAIN
            assert 0 <= o["xp_pct"] <= 100
        assert a["missed_better_option"] == any(o["better_option"] for o in a["options"])
        found += a["missed_better_option"]
    assert found > 0


def test_find_missed_passes_sorted(match):
    r = find_missed_passes(match, limit=5)
    gains = [x["gain_pct"] for x in r]
    assert gains == sorted(gains, reverse=True)


def test_bad_inputs(match):
    with pytest.raises(ValueError):
        analyse_pass(match, -1)
    p = match.passes[0]
    with pytest.raises(ValueError):
        pass_arrival(match, p["id"], p["player"])


def test_offside_rule():
    from whatelse.generator import is_offside
    defenders = np.array([[100, 34], [80, 30], [78, 40]])      # keeper at 100, second-last defender at 80
    ball = np.array([60, 34])
    assert is_offside(True, np.array([85, 34]), ball, defenders)        # beyond the second-last defender
    assert not is_offside(True, np.array([79, 34]), ball, defenders)    # level-ish but behind him
    assert not is_offside(True, np.array([45, 34]), np.array([40, 34]), np.array([[100, 34], [44, 30]]))  # own half


def test_xp_rises_with_margin():
    assert tools.completion_probability(-1.0) < tools.completion_probability(0.0) < tools.completion_probability(0.5)
    assert 0.6 < tools.completion_probability(0.0) < 0.8
