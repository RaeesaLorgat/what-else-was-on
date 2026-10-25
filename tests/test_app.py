import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from whatelse.generator import MatchSim  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    folder = tmp_path_factory.mktemp("data")
    MatchSim(seed=5, minutes=6).run().save(folder)
    mp = pytest.MonkeyPatch()
    mp.setenv("WHATELSE_DATA", str(folder))
    import whatelse.app as app_module
    app_module = importlib.reload(app_module)
    yield TestClient(app_module.app), folder
    mp.undo()


def test_lists_matches_and_moments(client):
    c, _ = client
    assert [m["seed"] for m in c.get("/api/matches").json()] == [5]
    m = c.get("/api/matches/5").json()
    assert m["moments"] and m["teams"][0]["name"]
    assert c.get("/api/matches/99").status_code == 404


def test_moment_has_clip_cone_and_options(client):
    c, _ = client
    pid = c.get("/api/matches/5").json()["moments"][0]["pass_id"]
    mo = c.get(f"/api/matches/5/moments/{pid}").json()
    clip = mo["clip"]
    assert len(clip["players"]) == len(clip["ball"]) and len(clip["players"][0]) == 22
    assert 0 <= clip["pass_index"] < len(clip["players"])
    best = [o for o in mo["analysis"]["options"] if o["better_option"]][0]
    assert len(best["aim"]) == 2 and best["closest_point"] is not None
    assert mo["story"] is None
    assert c.get("/api/matches/5/moments/-5").status_code == 404


def test_run_saves_the_story(client, monkeypatch):
    c, folder = client
    pid = c.get("/api/matches/5").json()["moments"][0]["pass_id"]
    import whatelse.llm as llm
    import whatelse.pipeline as pipeline

    async def fake_run(m, pass_id, client=None, checker_client=None):
        return {"pass_id": pass_id, "status": "approved", "verdict": {"time": "0:00"}, "commentary": "c",
                "pundit": "p", "drafts": 1, "trail": []}
    monkeypatch.setattr(llm, "load_settings", lambda: None)       # never call Azure from the tests
    monkeypatch.setattr(pipeline, "run_pass", fake_run)
    story = c.post(f"/api/matches/5/moments/{pid}/run").json()
    assert story["commentary"] == "c"
    assert (folder / "stories_5.json").exists()
    assert c.get(f"/api/matches/5/moments/{pid}").json()["story"]["status"] == "approved"


def test_page_is_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "What Else Was On" in r.text
    assert c.get("/static/app.js").status_code == 200


def test_live_runs_can_be_limited(client, monkeypatch):
    c, _ = client
    import whatelse.llm as llm
    monkeypatch.setattr(llm, "load_settings", lambda: None)
    monkeypatch.setenv("WHATELSE_LIVE_RUNS", "0")
    assert c.get("/api/config").json()["live_runs"] is False
    pid = c.get("/api/matches/5").json()["moments"][0]["pass_id"]
    assert c.post(f"/api/matches/5/moments/{pid}/run").status_code == 429
    monkeypatch.setenv("WHATELSE_LIVE_RUNS", "5")
    assert c.get("/api/config").json()["live_runs_left"] == 5
