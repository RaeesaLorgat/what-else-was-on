"""Checks in plain code, used by the Checker agent and the pipeline.

The language models write the words; this module makes sure every number
in those words is true. It recomputes the pass with the football tools and
compares field by field, so a verdict can only be approved if the tools agree.
"""

from __future__ import annotations

import re

from .tools import Match, analyse_pass

CLOSE_CALL_S = 0.3        # a pass that beats the nearest defender by less than this is "tight"
EVIDENCE_FIELDS = {       # evidence key -> option field in analyse_pass
    "better_option_threat_pct": "threat_pct",
    "gain_pct": "gain_vs_pass_played_pct",
    "better_option_in_view": "in_view",
    "better_option_offside": "offside",
    "better_option_arrives": "arrives",
    "better_option_xp_pct": "xp_pct",
    "closest_call_s": "closest_call_s",
}
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a == b


def numbers_in(text: str) -> list[str]:
    """Numbers written in a sentence, ignoring shirt numbers (#6) and match times (12:42)."""
    t = re.sub(r"#\d+", " ", text)
    t = re.sub(r"\b\d{1,3}:\d{2}\b", " ", t)
    return _NUM.findall(t)


def allowed_numbers(evidence: dict) -> set[float]:
    out = set()
    for v in evidence.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.add(round(abs(float(v)), 6))
    return out


def unsupported_numbers(text: str, evidence: dict, extra: set | None = None) -> list[str]:
    """Numbers in the text that are not in the evidence (extra = other allowed numbers, e.g. the match minute)."""
    ok = allowed_numbers(evidence) | {round(float(x), 6) for x in (extra or set())}
    return [n for n in numbers_in(text) if round(float(n), 6) not in ok]


def minutes_of(*times: str) -> set:
    """'75:05' -> {75, 76}: writers may say 'the 75th minute' or 'the 76th minute'."""
    out = set()
    for t in times:
        try:
            mm = int(str(t).split(":")[0])
            out |= {mm, mm + 1}
        except ValueError:
            pass
    return out


def facts_for(m: Match, pass_id: int, option_label: str | None) -> dict:
    """The tool numbers a verdict about this pass and option should contain."""
    a = analyse_pass(m, pass_id)
    opt = next((o for o in a["options"] if o["label"] == option_label), None) if option_label else None
    f = {"pass_id": pass_id, "time": a["time"], "passer": a["passer"], "pass_played_to": a["pass_played_to"],
         "missed_better_option": a["missed_better_option"], "best_better_option": a["best_better_option"],
         "better_options": [o["label"] for o in a["options"] if o["better_option"]],
         "threat_of_pass_played_pct": a["threat_of_pass_played_pct"], "option": opt}
    if opt:
        f["close_call"] = abs(opt["closest_call_s"]) < CLOSE_CALL_S
    return f


def verify_verdict(m: Match, v: dict) -> dict:
    """Compare an Analyst verdict with the tools. Returns the problems found (empty = numbers all correct)."""
    problems = []
    try:
        pid = int(v["pass_id"])
        f = facts_for(m, pid, v.get("better_option"))
    except (KeyError, TypeError, ValueError) as e:
        return {"problems": [f"verdict could not be checked: {e}"], "facts": None}

    for k in ("time", "passer", "pass_played_to"):
        if v.get(k) != f[k]:
            problems.append(f"{k} is {v.get(k)!r} but the tools say {f[k]!r}")
    ev = v.get("evidence") or {}
    if not _same(ev.get("threat_of_pass_played_pct"), f["threat_of_pass_played_pct"]):
        problems.append(f"threat_of_pass_played_pct is {ev.get('threat_of_pass_played_pct')} "
                        f"but the tools say {f['threat_of_pass_played_pct']}")

    if f["missed_better_option"]:
        if v.get("verdict") != "missed_better_option":
            problems.append(f"verdict should be missed_better_option: the tools found {', '.join(f['better_options'])}")
        elif v.get("better_option") not in f["better_options"]:
            problems.append(f"better_option {v.get('better_option')!r} is not one of the tools' better options "
                            f"({', '.join(f['better_options'])})")
        elif f["option"]:
            for ek, ok in EVIDENCE_FIELDS.items():
                if not _same(ev.get(ek), f["option"][ok]):
                    problems.append(f"{ek} is {ev.get(ek)} but the tools say {f['option'][ok]}")
    else:
        if v.get("verdict") != "good_choice":
            problems.append("verdict should be good_choice: the tools found no better option")
        if v.get("better_option") is not None:
            problems.append("better_option should be null when the verdict is good_choice")

    bad = unsupported_numbers(v.get("explanation", ""), ev, minutes_of(f["time"]))
    if bad:
        problems.append(f"the explanation uses numbers that are not in the evidence: {', '.join(bad)}")
    problems += [f"the explanation {p}" for p in wording_problems(v.get("explanation", ""), ev)]
    if f["missed_better_option"]:
        problems += [f"the explanation: {p}" for p in tight_problems(v.get("explanation", ""), ev.get("closest_call_s"))]
    return {"problems": problems, "facts": f}


# phrases that claim things the tools never measure (space, safety, goals)
BANNED_PHRASES = ["wide open", "wide-open", "clear path", "unmarked", "free in space", "acres of space",
                  "safer", "higher chance of success", "more likely to succeed", "would have scored", "easy pass",
                  "certain", "guaranteed", "tighter"]
_TIGHT = re.compile(r"\b(tight|close call|narrow(?:ly)?|risky|razor|only just)\b")
_NEGATION = re.compile(r"\b(not|without|no|never|wasn't|isn't|rather than)\b[^.]{0,30}$")


def tight_problems(text: str, closest_call_s) -> list[str]:
    """Tight means the pass would have beaten the nearest defender by less than CLOSE_CALL_S seconds."""
    if not isinstance(closest_call_s, (int, float)) or isinstance(closest_call_s, bool):
        return []
    low = text.lower()
    hits = [mt for mt in _TIGHT.finditer(low) if not _NEGATION.search(low[max(0, mt.start() - 40):mt.start()])]
    if closest_call_s < CLOSE_CALL_S and not hits:
        return [f"closest_call_s is {closest_call_s} (under {CLOSE_CALL_S} s): say the pass would have been tight"]
    if closest_call_s >= CLOSE_CALL_S and hits:
        return [f"closest_call_s is {closest_call_s} ({CLOSE_CALL_S} s or more): do not call it "
                f"'{hits[0].group(0)}'; it would have been comfortable"]
    return []
_COMPLETION_WORDS = ("complet", "come off", "comes off", "coming off", "succeed", "success", "connect", "find him",
                     "reach", "arrive", "get there", "pass")


def wording_problems(text: str, evidence: dict) -> list[str]:
    """Wording rules checked in code: no claims the tools do not measure, and the completion chance must be
    described as a completion chance (not a chance of scoring or of 'impact')."""
    low = text.lower()
    problems = [f"says '{ph}', which the tools do not measure" for ph in BANNED_PHRASES if ph in low]
    xp = evidence.get("better_option_xp_pct")
    if isinstance(xp, (int, float)):
        for mt in re.finditer(rf"(?<![\d.]){int(xp)}(?:\.0)?\s*(?:%|percent)", low):
            window = low[max(0, mt.start() - 60): mt.end() + 60]
            if not any(w in window for w in _COMPLETION_WORDS):
                problems.append(f"{int(xp)}% is the chance the pass would be completed; say so next to the number")
    return problems


def plain_facts(v: dict) -> str:
    """The verdict as plain sentences, so writers cannot misread what each number means."""
    ev = v.get("evidence") or {}
    if v.get("verdict") != "missed_better_option":
        return f"- {v.get('passer')} passed to {v.get('pass_played_to')}; nothing clearly better was on."
    cc = ev.get("closest_call_s")
    tight = (f"It would have beaten the nearest defender by only {cc} s, so it would have been TIGHT."
             if cc is not None and cc < CLOSE_CALL_S else
             f"It would have beaten the nearest defender by {cc} s, which is comfortable: do NOT call it tight.")
    return "\n".join([
        f"- At {v.get('time')}, {v.get('passer')} passed to {v.get('pass_played_to')}. "
        f"Threat where the ball went: {ev.get('threat_of_pass_played_pct')}%.",
        f"- The better option, NOT played: {v.get('better_option')}, in the passer's view and onside. "
        f"Threat there: {ev.get('better_option_threat_pct')}%, which is {ev.get('gain_pct')} points more.",
        f"- Had it been played, the pass to {v.get('better_option')} had a {ev.get('better_option_xp_pct')}% "
        "chance of being completed. This is a completion chance, not a chance of scoring or of impact. "
        "The completion chance of the pass actually played is not given, so do not compare safety.",
        f"- {tight}",
        "- Nothing is known about how open players were, their positions, runs, or what defenders did.",
    ])


def verdict_from_tools(m: Match, pass_id: int) -> dict:
    """A plain verdict written straight from the tools, used if the agents cannot agree."""
    a = analyse_pass(m, pass_id)
    best = next((o for o in a["options"] if o["better_option"]), None)
    ev = {"threat_of_pass_played_pct": a["threat_of_pass_played_pct"]}
    for ek, ok in EVIDENCE_FIELDS.items():
        ev[ek] = best[ok] if best else None
    if best:
        tight = " It would have been tight." if abs(best["closest_call_s"]) < CLOSE_CALL_S else ""
        text = (f"{a['passer']} played it to {a['pass_played_to']}, but {best['label']} was in view and would have "
                f"raised the threat from {a['threat_of_pass_played_pct']}% to {best['threat_pct']}%, "
                f"with a {best['xp_pct']}% chance of completing.{tight}")
    else:
        text = f"{a['passer']} found {a['pass_played_to']}; nothing clearly better was on."
    return {"pass_id": pass_id, "time": a["time"],
            "verdict": "missed_better_option" if best else "good_choice",
            "passer": a["passer"], "pass_played_to": a["pass_played_to"],
            "better_option": best["label"] if best else None, "evidence": ev, "explanation": text}
