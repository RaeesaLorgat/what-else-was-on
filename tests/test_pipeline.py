import asyncio
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("agent_framework")

from agent_framework.openai import OpenAIChatCompletionClient  # noqa: E402

import mock_model  # noqa: E402
from whatelse.checks import numbers_in, verdict_from_tools, verify_verdict  # noqa: E402
from whatelse.generator import MatchSim, load_match  # noqa: E402
from whatelse.pipeline import run_pass, write_recap  # noqa: E402
from whatelse.tools import Match, find_missed_passes  # noqa: E402


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    folder = tmp_path_factory.mktemp("m")
    MatchSim(seed=5, minutes=6).run().save(folder)
    return Match(*load_match(5, folder))


@pytest.fixture
def client():
    srv, url = mock_model.start()
    mock_model._Handler.sloppy = False
    yield OpenAIChatCompletionClient(model="mock-deployment", api_key="test", base_url=url)
    mock_model._Handler.sloppy = False
    srv.shutdown()


# ---------- plain-code checks ----------
def test_numbers_in_ignores_shirts_and_times():
    assert numbers_in("At 12:42 #11 Kowal found #7, but #6 had a 77% chance and +4.1") == ["77", "4.1"]


def test_tool_verdict_passes_its_own_check(match):
    pid = find_missed_passes(match, limit=1)[0]["pass_id"]
    v = verdict_from_tools(match, pid)
    assert verify_verdict(match, v)["problems"] == []


def test_check_catches_wrong_numbers_and_labels(match):
    pid = find_missed_passes(match, limit=1)[0]["pass_id"]
    good = verdict_from_tools(match, pid)
    bad = copy.deepcopy(good)
    bad["evidence"]["gain_pct"] += 1.0
    bad["explanation"] += " That is a 99% chance."
    probs = verify_verdict(match, bad)["problems"]
    assert any("gain_pct" in p for p in probs) and any("99" in p for p in probs)
    bad = copy.deepcopy(good)
    bad["verdict"], bad["better_option"] = "good_choice", None
    assert any("missed_better_option" in p for p in verify_verdict(match, bad)["problems"])


def test_resolve_teammate_accepts_label_shirt_or_id(match):
    from whatelse.tools import resolve_teammate, analyse_pass
    pid = find_missed_passes(match, limit=1)[0]["pass_id"]
    o = analyse_pass(match, pid)["options"][0]
    shirt = o["label"].split()[0]
    assert resolve_teammate(match, pid, o["label"]) == o["player"]
    assert resolve_teammate(match, pid, shirt) == o["player"]
    assert resolve_teammate(match, pid, str(o["player"])) == o["player"]
    with pytest.raises(ValueError):
        resolve_teammate(match, pid, "#99 Nobody")


def test_match_minute_is_allowed(match):
    pid = find_missed_passes(match, limit=1)[0]["pass_id"]
    v = verdict_from_tools(match, pid)
    minute = int(v["time"].split(":")[0])
    v["explanation"] += f" That was in the {minute + 1}th minute."
    assert verify_verdict(match, v)["problems"] == []


# ---------- the workflow, on the mock model ----------
def test_pipeline_approves_a_correct_verdict(match, client):
    pid = find_missed_passes(match, limit=1)[0]["pass_id"]
    story = asyncio.run(run_pass(match, pid, client=client))
    assert story["status"] == "approved" and story["drafts"] == 1
    assert story["verdict"]["better_option"] in story["pundit"]
    assert [t["agent"] for t in story["trail"]] == ["Analyst", "Checker", "Checker", "Narrator"]
    assert story["trail"][2]["on"] == "Narrator" and story["trail"][2]["approved"]


def test_checker_sends_a_wrong_verdict_back(match, client):
    mock_model._Handler.sloppy = True             # the Analyst's first draft has a wrong gain
    pid = find_missed_passes(match, limit=1)[0]["pass_id"]
    story = asyncio.run(run_pass(match, pid, client=client))
    assert story["status"] == "approved" and story["drafts"] == 2
    first_check = story["trail"][1]
    assert first_check["agent"] == "Checker" and not first_check["approved"]
    assert any("gain_pct" in p for p in first_check["problems"])
    assert verify_verdict(match, story["verdict"])["problems"] == []


def test_recap(match, client):
    ids = [r["pass_id"] for r in find_missed_passes(match, limit=2)]
    stories = [asyncio.run(run_pass(match, i, client=client)) for i in ids]
    r = asyncio.run(write_recap(stories, client=client))
    assert all(s["verdict"]["time"] in r["recap"] for s in stories)
    assert r["trail"][-1]["agent"] == "Checker" and r["trail"][-1]["approved"]


def test_wording_rules():
    from whatelse.checks import wording_problems
    ev = {"better_option_xp_pct": 80}
    assert wording_problems("Kowal was on, with an 80% chance to make a bigger impact.", ev)
    assert wording_problems("Kowal was wide open.", ev)
    assert wording_problems("Kowal was the safer ball.", ev)
    assert not wording_problems("Kowal was on: an 80% chance that pass comes off, though it would have been tight.", ev)


def test_tight_rule():
    from whatelse.checks import tight_problems
    assert tight_problems("That pass would have been tight.", 0.14) == []
    assert tight_problems("A lovely ball.", 0.14)
    assert tight_problems("It would have been quite tight.", 0.37)
    assert tight_problems("It would have arrived without being a tight call.", 0.37) == []


def test_real_problems_drops_fine_items():
    from whatelse.pipeline import real_problems
    items = ["Calling the 12:42 pass tight is correct (OK)", "'doubled the threat' is not supported",
             "wording 'tight chance' is ambiguous; acceptable"]
    assert real_problems(items) == ["'doubled the threat' is not supported"]
