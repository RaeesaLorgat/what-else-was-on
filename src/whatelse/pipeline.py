"""The three agents wired together as a Microsoft Agent Framework workflow.

    pass id --> Analyst --> Checker --approved--> Narrator --> story
                   ^           |
                   +-rejected--+   (the Checker's problems go back to the Analyst, at most 3 drafts)

If the Analyst still has not produced a verdict the Checker accepts after 3 drafts, the
pipeline does not guess: it uses a plain verdict written straight from the tools
(checks.verdict_from_tools) and marks the story as "fallback". The Narrator's words are
checked too: any number that is not in the approved verdict sends it back once, then
falls back to a plain line.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from typing_extensions import Never

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from .agents import (make_analyst, make_checker, make_copy_checker, make_client, make_narrator, make_recap_writer,
                     parse_json)
from .checks import (CLOSE_CALL_S, minutes_of, plain_facts, unsupported_numbers, verdict_from_tools, verify_verdict,
                     tight_problems, wording_problems)
from .tools import Match

MAX_DRAFTS = 3


@dataclass
class PassJob:
    pass_id: int


@dataclass
class Draft:
    pass_id: int
    verdict: dict
    attempt: int
    trail: list = field(default_factory=list)


@dataclass
class Review:
    draft: Draft
    approved: bool
    problems: list
    route: str          # "narrator" or "analyst"


_NOT_A_PROBLEM = re.compile(r"\b(no issue|no problem|acceptable|is correct|is fine|is accurate|\(ok\)|this is ok)\b",
                            re.I)


def real_problems(items) -> list[str]:
    """The model sometimes lists things it checked and found fine; keep only the real problems."""
    return [str(p) for p in (items or []) if not _NOT_A_PROBLEM.search(str(p))]


def _parse(text: str) -> dict:
    try:
        return parse_json(text)
    except (ValueError, json.JSONDecodeError):
        return {"unparsed_reply": text}


class AnalystStep(Executor):
    def __init__(self, m: Match, client):
        super().__init__(id="analyst")
        self.m, self.agent = m, make_analyst(m, client)

    @handler
    async def first(self, job: PassJob, ctx: WorkflowContext[Draft]) -> None:
        res = await self.agent.run(f"Analyse pass {job.pass_id}.")
        v = _parse(res.text)
        await ctx.send_message(Draft(job.pass_id, v, 1, [{"agent": "Analyst", "draft": 1, "verdict": v}]))

    @handler
    async def redo(self, r: Review, ctx: WorkflowContext[Draft]) -> None:
        d = r.draft
        prompt = (f"Analyse pass {d.pass_id}. Your previous verdict was rejected by the Checker.\n"
                  f"Previous verdict:\n{json.dumps(d.verdict)}\n"
                  f"Problems to fix:\n- " + "\n- ".join(r.problems) +
                  "\nCall the tools again if you need to, and reply with the corrected JSON only.")
        res = await self.agent.run(prompt)
        v = _parse(res.text)
        n = d.attempt + 1
        await ctx.send_message(Draft(d.pass_id, v, n, d.trail + [{"agent": "Analyst", "draft": n, "verdict": v}]))


class CheckerStep(Executor):
    def __init__(self, m: Match, client):
        super().__init__(id="checker")
        self.m, self.agent = m, make_checker(m, client)

    @handler
    async def check(self, d: Draft, ctx: WorkflowContext[Review]) -> None:
        code = verify_verdict(self.m, d.verdict) if "pass_id" in d.verdict else \
            {"problems": ["the Analyst did not reply with a JSON verdict"]}
        res = await self.agent.run("Check this verdict:\n" + json.dumps(d.verdict))
        said = _parse(res.text)
        problems = list(code["problems"])
        for p in real_problems(said.get("problems")):
            if p not in problems:
                problems.append(p)
        if "approved" not in said:
            problems.append("the Checker did not reply with a JSON decision")
        approved = not problems and (said.get("approved") is True or "approved" in said)
        route = "narrator" if approved or d.attempt >= MAX_DRAFTS else "analyst"
        d.trail.append({"agent": "Checker", "draft": d.attempt, "approved": approved, "problems": problems})
        await ctx.send_message(Review(d, approved, problems, route))


async def checked_copy(writer, checker, verdicts: list[dict], first_prompt: str, keys: tuple[str, ...],
                       trail: list, tries: int = 3) -> dict | None:
    """Ask the writer for copy, then check it twice: numbers in plain code, then the wording with the Checker.
    Rejected copy goes back to the writer with the problems. Returns None if it never passes."""
    ev = {f"{k}_{i}": x for i, v in enumerate(verdicts) for k, x in (v.get("evidence") or {}).items()}
    minutes = minutes_of(*[v.get("time", "") for v in verdicts])
    prompt = first_prompt
    for _ in range(tries):
        out = _parse((await writer.run(prompt)).text)
        if not out.get(keys[0]):
            problems = [f"no {keys[0]} in the reply"]
        else:
            words = " ".join(str(out.get(k, "")) for k in keys)
            bad = unsupported_numbers(words, ev, minutes)
            problems = [f"uses numbers not in the verdict: {', '.join(bad)}"] if bad else []
            for v in verdicts:
                problems += [p for p in wording_problems(words, v.get("evidence") or {}) if p not in problems]
            if len(verdicts) == 1 and verdicts[0].get("verdict") == "missed_better_option":
                problems += tight_problems(words, (verdicts[0].get("evidence") or {}).get("closest_call_s"))
            if not problems:
                said = _parse((await checker.run(
                    "Approved verdict(s):\n" + json.dumps(verdicts) + "\n\nWhat the numbers mean:\n" +
                    "\n\n".join(plain_facts(v) for v in verdicts) + "\n\nCopy to check:\n" +
                    json.dumps({k: out.get(k, "") for k in keys}))).text)
                problems = real_problems(said.get("problems"))
                if "approved" not in said:
                    problems = ["the Checker did not reply with a JSON decision"]
        trail.append({"agent": "Checker", "on": writer.name, "approved": not problems, "problems": problems})
        if not problems:
            return {k: out.get(k, "") for k in keys}
        prompt = (first_prompt + "\nYour last reply was rejected by the Checker:\n- " + "\n- ".join(problems) +
                  "\nFix every problem and reply with the JSON only.")
    return None


class NarratorStep(Executor):
    def __init__(self, m: Match, client, checker_client=None):
        super().__init__(id="narrator")
        self.m, self.agent = m, make_narrator(client)
        self.checker = make_copy_checker(checker_client or client)

    @handler
    async def narrate(self, r: Review, ctx: WorkflowContext[Never, dict]) -> None:
        d = r.draft
        if r.approved:
            verdict, status = d.verdict, "approved"
        else:
            verdict, status = verdict_from_tools(self.m, d.pass_id), "fallback"
            d.trail.append({"agent": "Pipeline", "note": f"no approved verdict after {MAX_DRAFTS} drafts; "
                                                         "using the plain verdict from the tools"})
        copy = await checked_copy(self.agent, self.checker, [verdict], "Approved verdict:\n" + json.dumps(verdict) +
                                  "\n\nWhat the numbers mean:\n" + plain_facts(verdict),
                                  ("commentary", "pundit"), d.trail)
        if copy is None:
            copy = plain_copy(verdict)
            d.trail.append({"agent": "Pipeline", "note": "using a plain commentary line"})
        d.trail.append({"agent": "Narrator", "commentary": copy["commentary"]})
        await ctx.yield_output({"pass_id": d.pass_id, "status": status, "verdict": verdict, **copy,
                                "drafts": d.attempt, "trail": d.trail})


def plain_copy(v: dict) -> dict:
    ev = v.get("evidence") or {}
    if v.get("verdict") != "missed_better_option":
        return {"commentary": f"{v.get('passer')} keeps it simple to {v.get('pass_played_to')}.", "pundit": ""}
    tight = " It would have been tight." if (ev.get("closest_call_s") or 99) < CLOSE_CALL_S else ""
    return {"commentary": f"{v['passer']} goes to {v['pass_played_to']}, but {v['better_option']} was on.",
            "pundit": f"Look at {v['better_option']}: in view, and a {ev.get('better_option_xp_pct')}% chance "
                      f"that pass comes off.{tight}"}


def build_workflow(m: Match, client=None, checker_client=None):
    """client runs the Analyst and Narrator; checker_client runs the Checker (a stronger model if configured).
    If only client is given (e.g. the test stand-in), it runs everything."""
    if client is None:
        client, checker_client = make_client(), checker_client or make_client("checker")
    checker_client = checker_client or client
    analyst = AnalystStep(m, client)
    checker = CheckerStep(m, checker_client)
    narrator = NarratorStep(m, client, checker_client)
    return (WorkflowBuilder(start_executor=analyst, name="what-else-was-on")
            .add_edge(analyst, checker)
            .add_edge(checker, narrator, condition=lambda r: r.route == "narrator")
            .add_edge(checker, analyst, condition=lambda r: r.route == "analyst")
            .build())


async def run_pass(m: Match, pass_id: int, client=None, checker_client=None) -> dict:
    """Analyst -> Checker (-> Analyst ...) -> Narrator for one pass. Returns the story with its full trail."""
    wf = build_workflow(m, client, checker_client)
    res = await wf.run(PassJob(pass_id))
    outs = res.get_outputs()
    if not outs:
        raise RuntimeError(f"the workflow ended without a story for pass {pass_id}")
    return outs[-1]


async def write_recap(stories: list[dict], client=None, checker_client=None) -> dict:
    """Post-match recap from several approved stories, checked like everything else.
    Returns {"recap": text, "trail": [...]}."""
    verdicts = [s["verdict"] for s in stories if s["verdict"].get("verdict") == "missed_better_option"]
    if not verdicts:
        return {"recap": "No clear missed passes in this match.", "trail": []}

    def clock(v):
        mm, ss = (str(v.get("time", "0:0")) + ":0").split(":")[:2]
        return int(mm) * 60 + int(ss)
    verdicts.sort(key=clock)                  # tell the match in order
    if client is None:
        client, checker_client = make_client(), checker_client or make_client("checker")
    checker_client = checker_client or client
    trail: list = []
    # the recap is the longest piece of writing, so it uses the stronger (Checker) model too
    out = await checked_copy(make_recap_writer(checker_client), make_copy_checker(checker_client), verdicts,
                             "Approved verdicts:\n" + json.dumps(verdicts) + "\n\nWhat the numbers mean:\n" +
                             "\n\n".join(plain_facts(v) for v in verdicts), ("recap",), trail)
    if out is None:
        out = {"recap": " ".join(f"At {v['time']}, {v['passer']} went to {v['pass_played_to']} when "
                                 f"{v['better_option']} was on." for v in verdicts)}
        trail.append({"agent": "Pipeline", "note": "using a plain recap"})
    return {"recap": out["recap"], "trail": trail}
