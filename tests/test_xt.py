import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whatelse import xt  # noqa: E402

# a made-up grid in the same layout as the real file: 8 rows, 12 columns, first row = far touchline
FAKE = [[0.01 * (c + 1) + 0.001 * r for c in range(12)] for r in range(8)]


@pytest.fixture
def fake_grid():
    xt.set_grid(FAKE)
    yield
    xt._loaded = False          # next call reloads from data/ (or uses the hand-made map)
    xt._grid = None


def test_cell_centres_match_the_file(fake_grid):
    # centre of the first column, nearest touchline (y small) = last row of the file
    assert xt.xt(105 / 24, 68 / 16, True) == pytest.approx(FAKE[7][0])
    # centre of the last column, far touchline = first row of the file
    assert xt.xt(105 - 105 / 24, 68 - 68 / 16, True) == pytest.approx(FAKE[0][11])


def test_interpolates_between_cells(fake_grid):
    a = xt.xt(105 / 24, 30, True)
    b = xt.xt(105 / 24 + 105 / 12, 30, True)
    mid = xt.xt(105 / 24 + 105 / 24, 30, True)
    assert mid == pytest.approx((a + b) / 2)


def test_other_team_attacks_the_other_way(fake_grid):
    assert xt.xt(100, 20, True) == pytest.approx(xt.xt(5, 48, False))
    assert "Karun" in xt.model_name()


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        xt.check_grid([[0.1] * 8] * 12)


def test_falls_back_without_grid():
    xt.set_grid(None)
    try:
        assert xt.model_name() == "hand-made xT-style map"
        assert 0 < xt.xt(100, 34, True) < 1
    finally:
        xt._loaded = False
