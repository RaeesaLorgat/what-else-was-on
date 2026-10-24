"""The agents, built with Microsoft Agent Framework on a model deployed in Microsoft Foundry.

- Analyst:  given a pass, calls the football tools, decides whether something better
            was on, and returns a short, evidence-backed verdict as JSON.
- Checker:  re-runs the tools on the Analyst's verdict (verify_verdict, plain code)
            and reads the wording; it rejects anything the numbers do not support.
- Narrator: turns an approved verdict into a live commentary line and a pundit line,
            and several verdicts into a short post-match recap.

pipeline.py wires them together as an Agent Framework workflow.
"""

from __future__ import annotations

import json
import re
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient

from .llm import load_settings
from .checks import verify_verdict
from .tools import Match, analyse_pass, find_missed_passes, offside_check, pass_arrival, resolve_teammate

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
}

Writing the explanation:
- The better option was NOT played, so talk about it with "would have" / "could have", never as if it happened.
- If closest_call_s is below 0.3, the pass would only just have beaten the defender: say it would have been tight.
  If it is 0.3 or more, do not use the words tight, close call, narrow or risky at all (you may say comfortable).
- Do not claim more than the numbers show (no "would have scored", no "easy").
- Only use numbers from the evidence object. Shirt numbers and the match time are fine.

If you are told the Checker rejected an earlier verdict, fix every problem it lists and reply with the corrected JSON."""

CHECKER_INSTRUCTIONS = """You are the Checker in a football broadcast team. You never trust the Analyst: you verify.
You receive the Analyst's verdict on one pass as JSON.

1. Call verify_verdict with the verdict JSON exactly as given. It recomputes the pass with the football
   tools and lists every number or label that does not match. Each listed problem is a reason to reject.
2. Then read the explanation yourself, using the facts that verify_verdict returns, and reject it if:
   - it talks about the better option as if the pass was actually played ("arrived", "found him") instead of
     "would have" ("would have arrived" is correct);
   - it claims something the numbers do not show (a goal, "easy", "unmarked", "certain");
   - it uses jargon a TV viewer would not know (xT, xP, threat model, percentage points is fine).
   Whether the pass was tight is checked in code by verify_verdict (tight means closest_call_s below 0.3);
   do not judge tightness yourself, and never add a problem about it.
3. List ONLY real problems; never list something that is fine or correct. Approve if there are none.

Reply with ONLY a JSON object:
{"approved": true or false, "problems": ["<short, specific problem>", ...]}"""

NARRATOR_INSTRUCTIONS = """You are the Narrator for a live football broadcast. You receive a verdict about one pass
that the Checker has already approved, plus "What the numbers mean" in plain sentences. Turn it into broadcast copy,
following those sentences exactly.

Rules:
- Use only facts and numbers from the verdict. Do not add any number that is not in it.
- The better option was not played: say what was on, not what happened.
- If closest_call_s is below 0.3, make it clear the pass would have been tight. If it is 0.3 or more, do not use
  tight, close call, narrow or risky at all. Never write "tighter".
- Do not describe anything the verdict does not say (where players were standing, runs, what defenders did).
- When you give the completion chance, say it is the chance the pass would come off / be completed.
- The verdict gives the completion chance of the better option only, not of the pass played. So never say the
  better option was safer or had a "higher chance of success"; it was the more dangerous option.
- Plain words, no jargon.

Reply with ONLY a JSON object:
{"commentary": "<one live commentary line, at most 25 words>",
 "pundit": "<one or two sentences a studio pundit would say over the replay>"}"""

COPY_CHECK_INSTRUCTIONS = """You are the Checker in a football broadcast team, now checking broadcast copy
(commentary, a pundit line or a recap) against the approved verdict(s) it is based on. Be strict but fair.

Reject the copy if any sentence:
- says something the verdicts do not support, e.g. the better option was safer or had a higher chance of
  success (the verdict only gives the better option's completion chance, not the pass played's), defenders'
  actions, player positions, or that a goal would have come;
- calls the passer's decision or the pass actually played "tight" (tight only ever describes the pass that was not
  played); whether that pass was tight is already checked in code, so do not judge the threshold yourself;
- describes the better option as if it was actually played;
- is muddled or contradicts itself.
Small style choices are fine. Do not reject for missing details. Whether a pass was tight is checked in code.
List ONLY real problems. Never list something that is fine, correct or acceptable; if everything is fine,
return an empty list and approve.

Reply with ONLY a JSON object:
{"approved": true or false, "problems": ["<short, specific problem, quoting the words>", ...]}"""

RECAP_INSTRUCTIONS = """You are the Narrator for a football broadcast. You receive a list of approved verdicts
from one match, each about a pass where something better was on. Write the post-match recap segment.

Rules:
- At most 120 words, plain words, no jargon.
- Mention each moment by its match time and players. Use only numbers that appear in the verdicts.
- The better options were not played: talk about what was on, not what happened.
- Never say a better option was safer or more likely to succeed; it was the more dangerous option.
- If a moment's closest_call_s is below 0.3, say that pass would have been tight; never call the others tight,
  never call the pass actually played tight, and never write "tighter".

Reply with ONLY a JSON object: {"recap": "<text>"}"""


def make_client(role: str = "default"):
    """Agent Framework chat client pointed at the Foundry deployment from .env.
    role="checker" uses AZURE_OPENAI_CHECKER_MODEL if it is set (a stronger model for the Checker)."""
    s = load_settings()
    model = s.checker_model if role == "checker" else s.model
    return OpenAIChatCompletionClient(model=model, api_key=s.api_key, base_url=s.base_url)


def make_tools(m: Match) -> list:
    """The football tools, bound to one match, wrapped so the agent can call them."""

    @tool(name="analyse_pass", description="Full analysis of one pass: every teammate with in_view, threat, "
                                           "gain vs the pass played, whether a pass would arrive, and better_option.")
    def analyse_pass_tool(pass_id: Annotated[int, "id of the pass event"]) -> str:
        return json.dumps(analyse_pass(m, pass_id))

    @tool(name="check_pass_arrival", description="Would a pass from this passer to the given teammate "
                                                 "arrive before any defender? Gives the closest call in seconds.")
    def check_pass_arrival(pass_id: Annotated[int, "id of the pass event"],
                           teammate: Annotated[str, "the teammate's label from analyse_pass, e.g. '#6 Ben Bjorn'"]) -> str:
        try:
            return json.dumps(pass_arrival(m, pass_id, resolve_teammate(m, pass_id, teammate)))
        except ValueError as e:
            return json.dumps({"error": str(e)})

    @tool(name="check_offside", description="Was this teammate offside at the moment of the pass?")
    def check_offside(pass_id: Annotated[int, "id of the pass event"],
                      teammate: Annotated[str, "the teammate's label from analyse_pass, e.g. '#6 Ben Bjorn'"]) -> str:
        try:
            return json.dumps(offside_check(m, pass_id, resolve_teammate(m, pass_id, teammate)))
        except ValueError as e:
            return json.dumps({"error": str(e)})

    @tool(name="list_missed_passes", description="Passes in this match where a better visible option was on, "
                                                 "biggest threat gain first.")
    def list_missed_passes(limit: Annotated[int, "how many to return"] = 5) -> str:
        return json.dumps(find_missed_passes(m, limit=limit))

    return [analyse_pass_tool, check_pass_arrival, check_offside, list_missed_passes]


def make_analyst(m: Match, client=None) -> Agent:
    return Agent(client=client or make_client(), name="Analyst", instructions=ANALYST_INSTRUCTIONS,
                 tools=make_tools(m))


def make_checker_tools(m: Match) -> list:
    @tool(name="verify_verdict", description="Recompute the pass with the football tools and list every number "
                                             "or label in the verdict that does not match. Also returns the facts.")
    def verify(verdict_json: Annotated[str, "the Analyst's verdict, as a JSON string"]) -> str:
        try:
            v = parse_json(verdict_json) if isinstance(verdict_json, str) else verdict_json
        except ValueError as e:
            return json.dumps({"problems": [f"not valid JSON: {e}"], "facts": None})
        return json.dumps(verify_verdict(m, v))

    return [verify]


def make_checker(m: Match, client=None) -> Agent:
    return Agent(client=client or make_client(), name="Checker", instructions=CHECKER_INSTRUCTIONS,
                 tools=make_checker_tools(m))


def make_narrator(client=None) -> Agent:
    return Agent(client=client or make_client(), name="Narrator", instructions=NARRATOR_INSTRUCTIONS)


def make_copy_checker(client=None) -> Agent:
    return Agent(client=client or make_client(), name="CopyChecker", instructions=COPY_CHECK_INSTRUCTIONS)


def make_recap_writer(client=None) -> Agent:
    return Agent(client=client or make_client(), name="Recap", instructions=RECAP_INSTRUCTIONS)


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
