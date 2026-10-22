"""The agents, built with Microsoft Agent Framework on a model deployed in Microsoft Foundry.

Day 5: the Analyst agent. Given a pass, it calls the football tools, decides
whether something better was on, and returns a short, evidence-backed verdict
as JSON. It may only use numbers that the tools gave it; the Checker agent
(next step) will verify that.
"""

from __future__ import annotations

import json
import re
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient

from .llm import load_settings
from .tools import Match, analyse_pass, find_missed_passes, offside_check, pass_arrival

ANALYST_INSTRUCTIONS = """You are a football performance analyst for a TV broadcast.
You review one pass at a time and answer: was something better on?

How to work:
1. Always call analyse_pass for the pass you are given. Never guess numbers.
2. A "better option" is defined by the tool (field better_option): a teammate in the passer's
   120 degree vision cone, not offside, at least 0.5 percentage points more threat than the pass played,
   and a pass that would arrive before any defender. Use that definition; do not invent your own.
   xp_pct is the chance that pass would be completed; mention it in the explanation.
3. If there is a better option, you may call check_pass_arrival for it to describe how close the nearest defender was.
4. Use ONLY numbers that appear in tool results, copied exactly. Round nothing.

Reply with ONLY a JSON object, no other text, in this shape:
{
  "pass_id": <int>,
  "time": "<mm:ss>",
  "verdict": "missed_better_option" or "good_choice",
  "passer": "<label>",
  "pass_played_to": "<label>",
  "better_option": "<label>" or null,
  "evidence": {
    "threat_of_pass_played_pct": <number>,
    "better_option_threat_pct": <number or null>,
    "gain_pct": <number or null>,
    "better_option_in_view": <true/false/null>,
    "better_option_offside": <true/false/null>,
    "better_option_arrives": <true/false/null>,
    "better_option_xp_pct": <number or null>,
    "closest_call_s": <number or null>
  },
  "explanation": "<two short sentences a TV analyst could say, no jargon>"
}"""


def make_client():
    """Agent Framework chat client pointed at the Foundry deployment from .env."""
    s = load_settings()
    return OpenAIChatCompletionClient(model=s.model, api_key=s.api_key, base_url=s.base_url)


def make_tools(m: Match) -> list:
    """The football tools, bound to one match, wrapped so the agent can call them."""

    @tool(name="analyse_pass", description="Full analysis of one pass: every teammate with in_view, threat, "
                                           "gain vs the pass played, whether a pass would arrive, and better_option.")
    def analyse_pass_tool(pass_id: Annotated[int, "id of the pass event"]) -> str:
        return json.dumps(analyse_pass(m, pass_id))

    @tool(name="check_pass_arrival", description="Would a pass from this passer to the given teammate (player id) "
                                                 "arrive before any defender? Gives the closest call in seconds.")
    def check_pass_arrival(pass_id: Annotated[int, "id of the pass event"],
                           teammate: Annotated[int, "player id of the teammate"]) -> str:
        return json.dumps(pass_arrival(m, pass_id, teammate))

    @tool(name="check_offside", description="Was this teammate (player id) offside at the moment of the pass?")
    def check_offside(pass_id: Annotated[int, "id of the pass event"],
                      teammate: Annotated[int, "player id of the teammate"]) -> str:
        return json.dumps(offside_check(m, pass_id, teammate))

    @tool(name="list_missed_passes", description="Passes in this match where a better visible option was on, "
                                                 "biggest threat gain first.")
    def list_missed_passes(limit: Annotated[int, "how many to return"] = 5) -> str:
        return json.dumps(find_missed_passes(m, limit=limit))

    return [analyse_pass_tool, check_pass_arrival, check_offside, list_missed_passes]


def make_analyst(m: Match, client=None) -> Agent:
    return Agent(client=client or make_client(), name="Analyst", instructions=ANALYST_INSTRUCTIONS,
                 tools=make_tools(m))


def parse_json(text: str) -> dict:
    """Pull the JSON object out of the agent's reply (tolerates ```json fences)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("The agent did not return JSON:\n" + text)
    return json.loads(t[start:end + 1])


async def analyse_with_agent(m: Match, pass_id: int, client=None) -> dict:
    agent = make_analyst(m, client)
    res = await agent.run(f"Analyse pass {pass_id}.")
    return parse_json(res.text)
