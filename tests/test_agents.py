import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("agent_framework")

from agent_framework.openai import OpenAIChatCompletionClient  # noqa: E402

import mock_model  # noqa: E402
from whatelse.agents import analyse_with_agent, parse_json  # noqa: E402
from whatelse.generator import MatchSim, load_match  # noqa: E402
from whatelse.tools import Match, analyse_pass, find_missed_passes  # noqa: E402


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    folder = tmp_path_factory.mktemp("m")
    MatchSim(seed=5, minutes=6).run().save(folder)
    return Match(*load_match(5, folder))


def test_parse_json_handles_fences():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('{"a": 2}') == {"a": 2}


def test_analyst_calls_tool_and_returns_tool_numbers(match):
    srv, url = mock_model.start()
    try:
        client = OpenAIChatCompletionClient(model="mock-deployment", api_key="test", base_url=url)
        pid = find_missed_passes(match, limit=1)[0]["pass_id"]
        out = asyncio.run(analyse_with_agent(match, pid, client=client))
    finally:
        srv.shutdown()
    truth = analyse_pass(match, pid)
    assert out["pass_id"] == pid and out["verdict"] == "missed_better_option"
    assert out["better_option"] == truth["best_better_option"]
    assert out["evidence"]["threat_of_pass_played_pct"] == truth["threat_of_pass_played_pct"]
    first = mock_model._Handler.requests[0]
    assert first["model"] == "mock-deployment"
    assert any(t["function"]["name"] == "analyse_pass" for t in first["tools"])
