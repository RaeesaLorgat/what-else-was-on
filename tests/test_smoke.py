import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test import distance_m  # noqa: E402


def test_distance_worked_example():
    # (40, 30) to (58, 22): dx = 18, dy = -8, sqrt(324 + 64) = 19.70 m
    assert distance_m(40, 30, 58, 22) == 19.7


def test_smoke_test_runs_in_mock_mode():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py"), "--mock"],
                         capture_output=True, text=True, check=True).stdout
    assert "All checks passed." in out
    assert "19.7" in out
