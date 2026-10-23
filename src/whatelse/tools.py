"""Football tools the agents call.

Each tool looks at one pass in a generated match and answers one question:

- vision_cone:   which teammates could the passer see?
- option_value:  how dangerous (expected threat, see xt.py) would the ball be at each teammate, compared
                 with where the pass actually went?
- pass_arrival:  would a pass to a teammate get there before a defender, and how
                 likely is it to be completed (xP)?
- offside_check: was the teammate offside at the moment of the pass?
- analyse_pass:  all three together, plus the verdict "was something better on?"
- find_missed_passes: the passes in a match where the best visible option was
                 clearly better than the pass played, and would have arrived.

Every tool returns plain dictionaries (JSON-ready), so an agent can read the
numbers and quote them.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .generator import load_match, is_offside, PITCH_X, PITCH_Y, HZ
from .xt import xt as threat, model_name as threat_model

CONE_HALF_ANGLE = 60.0     # degrees either side: a 120 degree vision cone
FACING_WINDOW = 0.5        # seconds of movement used to decide where the passer faces
BALL_SPEED = 12.5          # m/s, a typical ground pass
REACT = 0.7                # s, reaction time before a defender changes direction
VMAX = 7.0                 # m/s, top speed when chasing the ball
BETTER_GAIN = 0.005        # a better option must add at least 0.5 percentage points of threat (xT)
# on Karun Singh's grid, one zone forward in midfield adds about 0.2-0.5 points, so 0.5 means a clear step up;
# it flags about 1 pass in 10, close to the share flagged with the hand-made map at 1 point
# completion probability (xP) from the arrival margin: fitted by logistic regression on
# 4,851 passes from synthetic matches 1-3 (scripts/fit_xp.py); AUC 0.69
XP_INTERCEPT, XP_SLOPE = 0.979, 1.468
MIN_DIST, MAX_DIST = 4.0, 45.0   # passes shorter or longer than this are not considered


class Match:
    """A generated match, with helpers to look things up at any frame."""

    def __init__(self, meta: dict, trk: dict):
        self.meta = meta
        self.players = meta["players"]
        self.tracks = trk["players"].astype(float)
        self.ball = trk["ball"].astype(float)
        self.events = meta["events"]
        self.passes = [e for e in self.events if e["type"] == "pass"]
        self._by_id = {e["id"]: e for e in self.events}

    @classmethod
    def load(cls, seed: int, folder: str | Path = "data") -> "Match":
        return cls(*load_match(seed, folder))

    def event(self, event_id: int) -> dict:
        if event_id not in self._by_id or self._by_id[event_id]["type"] != "pass":
            raise ValueError(f"Event {event_id} is not a pass in this match.")
        return self._by_id[event_id]

    def pos(self, frame: int) -> np.ndarray:
        return self.tracks[min(max(frame, 0), len(self.tracks) - 1)]

    def vel(self, frame: int, window: float = 0.2) -> np.ndarray:
        k = max(1, int(round(window * HZ)))
        return (self.pos(frame) - self.pos(frame - k)) / (k / HZ)

    def team_of(self, pid: int) -> int:
        return self.players[pid]["team"]

    def attacks_right(self, team: int) -> bool:
        return self.meta["teams"][team]["attacks"] == "right"

    def label(self, pid: int) -> str:
        p = self.players[pid]
        return f"#{p['jersey']} {p['name']}"


@lru_cache(maxsize=8)
def _cached(seed: int, folder: str) -> Match:
    return Match.load(seed, folder)


def get_match(seed: int, folder: str | Path = "data") -> Match:
    return _cached(seed, str(folder))


# ---------- vision cone ----------
def facing_direction(m: Match, passer: int, frame: int) -> np.ndarray:
    """Unit vector of where the passer is looking: the way he moved in the last half second,
    or towards the goal he attacks if he was nearly standing still."""
    v = m.vel(frame, FACING_WINDOW)[passer]
    sp = float(np.hypot(*v))
    if sp > 0.8:
        return v / sp
    return np.array([1.0, 0.0]) if m.attacks_right(m.team_of(passer)) else np.array([-1.0, 0.0])


def vision_cone(m: Match, pass_id: int) -> dict:
    e = m.event(pass_id)
    f, i = e["frame"], e["player"]
    P = m.pos(f)
    face = facing_direction(m, i, f)
    mates = []
    for p in m.players:
        j = p["pid"]
        if j == i or p["team"] != m.team_of(i):
            continue
        d = P[j] - P[i]
        dist = float(np.hypot(*d))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(d / max(dist, 1e-6), face))))))
        mates.append({"player": j, "label": m.label(j), "in_view": ang <= CONE_HALF_ANGLE,
                      "angle_from_facing_deg": round(ang), "distance_m": round(dist, 1)})
    return {"pass_id": pass_id, "passer": m.label(i),
            "facing_deg": round(math.degrees(math.atan2(face[1], face[0]))),
            "cone_deg": 2 * CONE_HALF_ANGLE, "teammates": mates}


def lead_aim(m: Match, passer: int, target: int, frame: int) -> np.ndarray:
    """Where a pass to `target` should be aimed: where he will be when a ground pass gets there."""
    P, V = m.pos(frame), m.vel(frame)
    d0 = float(np.hypot(*(P[target] - P[passer])))
    aim = P[target] + V[target] * (d0 / BALL_SPEED)
    return np.clip(aim, [0.5, 0.5], [PITCH_X - 0.5, PITCH_Y - 0.5])


def completion_probability(closest_call_s: float) -> float:
    """xP: chance the pass is completed, from how much time the ball has to spare over the quickest defender."""
    m = max(-2.0, min(2.0, closest_call_s))
    return 1.0 / (1.0 + math.exp(-(XP_INTERCEPT + XP_SLOPE * m)))


# ---------- offside ----------
def offside(m: Match, passer: int, target: int, frame: int) -> bool:
    team = m.team_of(passer)
    opp = [p["pid"] for p in m.players if p["team"] != team]
    P = m.pos(frame)
    return is_offside(m.attacks_right(team), P[target], m.ball[min(frame, len(m.ball) - 1)], P[opp])


def resolve_teammate(m: Match, pass_id: int, ref) -> int:
    """Find the passer's teammate from a player id, a shirt number or a label like "#6 Ben Bjorn"."""
    passer = m.event(pass_id)["player"]
    team = m.team_of(passer)
    mates = [p for p in m.players if p["team"] == team and p["pid"] != passer]
    text = str(ref).strip()
    for p in mates:
        if text == m.label(p["pid"]) or text.lower() == p["name"].lower():
            return p["pid"]
    digits = text.lstrip("#").split()[0] if text.lstrip("#") else ""
    if digits.isdigit():
        n = int(digits)
        by_jersey = [p["pid"] for p in mates if p["jersey"] == n]
        if text.startswith("#") and by_jersey:
            return by_jersey[0]
        if any(p["pid"] == n for p in mates):
            return n
        if by_jersey:
            return by_jersey[0]
    raise ValueError(f"{ref!r} is not a teammate of {m.label(passer)}; use one of: "
                     + ", ".join(m.label(p["pid"]) for p in mates))


def offside_check(m: Match, pass_id: int, teammate: int) -> dict:
    e = m.event(pass_id)
    off = offside(m, e["player"], teammate, e["frame"])
    return {"pass_id": pass_id, "teammate": m.label(teammate), "offside": off,
            "rule": "offside if in the opponents' half, ahead of the ball and ahead of the second-last opponent"}


# ---------- option value ----------
def option_value(m: Match, pass_id: int) -> dict:
    e = m.event(pass_id)
    f, i = e["frame"], e["player"]
    team = m.team_of(i)
    right = m.attacks_right(team)
    played = threat(e["end_x"], e["end_y"], right)
    out = []
    for p in m.players:
        j = p["pid"]
        if j == i or p["team"] != team:
            continue
        aim = lead_aim(m, i, j, f)
        val = threat(float(aim[0]), float(aim[1]), right)
        out.append({"player": j, "label": m.label(j), "threat_pct": round(100 * val, 1),
                    "gain_vs_pass_played_pct": round(100 * (val - played), 1)})
    return {"pass_id": pass_id, "threat_of_pass_played_pct": round(100 * played, 1),
            "threat_model": threat_model(),
            "note": "threat = chance the move ends in a goal soon if the ball is at that spot", "options": out}


# ---------- pass arrival ----------
def _reach_time(pos: np.ndarray, vel: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Seconds each player needs to reach pt: keeps his momentum for REACT seconds, then sprints straight at VMAX."""
    q = pos + vel * REACT
    return REACT + np.hypot(*(pt - q).T) / VMAX


def arrival(m: Match, passer: int, target: int, frame: int) -> dict:
    P, V = m.pos(frame), m.vel(frame)
    start = P[passer]
    aim = lead_aim(m, passer, target, frame)                         # lead the pass into his run
    L = float(np.hypot(*(aim - start)))
    opp = np.array([p["pid"] for p in m.players if p["team"] != m.team_of(passer)])
    worst, cut = None, None
    for s in np.arange(2.0, L + 0.01, 1.0):
        pt = start + (aim - start) * s / max(L, 1e-6)
        t_ball = s / BALL_SPEED
        t_opp = _reach_time(P[opp], V[opp], pt)
        k = int(np.argmin(t_opp))
        spare = float(t_opp[k] - t_ball)              # >0: ball gets there first by this many seconds
        if worst is None or spare < worst[0]:
            worst = (spare, int(opp[k]), float(s))
        if cut is None and spare < 0:
            cut = (int(opp[k]), float(s))
    spare, who, where = worst if worst else (99.0, None, 0.0)
    res = {"arrives": cut is None, "ball_time_s": round(L / BALL_SPEED, 2),
           "closest_call_s": round(spare, 2), "closest_defender": m.label(who) if who is not None else None,
           "xp_pct": round(100 * completion_probability(spare))}
    if cut:
        res.update({"cut_out_by": m.label(cut[0]), "cut_out_after_m": round(cut[1])})
    res["close_call"] = abs(spare) < 0.3
    return res


def pass_arrival(m: Match, pass_id: int, teammate: int) -> dict:
    e = m.event(pass_id)
    if m.team_of(teammate) != m.team_of(e["player"]) or teammate == e["player"]:
        raise ValueError("teammate must be another player on the passer's team")
    r = arrival(m, e["player"], teammate, e["frame"])
    return {"pass_id": pass_id, "to": m.label(teammate), **r,
            "model": f"ground pass at {BALL_SPEED} m/s; defenders react after {REACT} s and sprint at {VMAX} m/s"}


# ---------- all together ----------
def analyse_pass(m: Match, pass_id: int) -> dict:
    """Everything an analyst needs about one pass: who was in view, how valuable each option was,
    whether it would have arrived, and which options were better than the pass played."""
    e = m.event(pass_id)
    i, f = e["player"], e["frame"]
    cone = {t["player"]: t for t in vision_cone(m, pass_id)["teammates"]}
    vals = option_value(m, pass_id)
    played_target = e["target"]
    options = []
    for o in vals["options"]:
        j = o["player"]
        c = cone[j]
        if not (MIN_DIST <= c["distance_m"] <= MAX_DIST):
            continue
        a = arrival(m, i, j, f)
        off = offside(m, i, j, f)
        better = (c["in_view"] and not off and o["gain_vs_pass_played_pct"] >= 100 * BETTER_GAIN and a["arrives"]
                  and j != played_target)
        options.append({"player": j, "label": o["label"], "in_view": c["in_view"], "offside": off,
                        "distance_m": c["distance_m"], "threat_pct": o["threat_pct"],
                        "gain_vs_pass_played_pct": o["gain_vs_pass_played_pct"],
                        "arrives": a["arrives"], "xp_pct": a["xp_pct"], "closest_call_s": a["closest_call_s"],
                        "close_call": a["close_call"],
                        "cut_out_by": a.get("cut_out_by"), "better_option": better,
                        "was_pass_played": j == played_target})
    options.sort(key=lambda o: -o["gain_vs_pass_played_pct"])
    better = [o for o in options if o["better_option"]]
    mins, secs = divmod(int(e["t"]), 60)
    return {
        "pass_id": pass_id, "time": f"{mins}:{secs:02d}",
        "team": m.meta["teams"][m.team_of(i)]["name"],
        "passer": m.label(i), "pass_played_to": m.label(played_target),
        "pass_outcome": e["outcome"],
        "pass_played_in_view": bool(e["extra"].get("in_view")),
        "threat_of_pass_played_pct": vals["threat_of_pass_played_pct"],
        "threat_model": vals["threat_model"],
        "missed_better_option": bool(better),
        "best_better_option": better[0]["label"] if better else None,
        "options": options,
        "rule": "better option = teammate in the 120 degree cone, not offside, at least 0.5 points more threat than "
                "the pass played, and the pass would arrive before any defender",
    }


def find_missed_passes(m: Match, limit: int = 10, team: int | None = None) -> list[dict]:
    """Passes where a visible, safe option was clearly more dangerous than the pass played, biggest gain first."""
    out = []
    for e in m.passes:
        if team is not None and e["team"] != team:
            continue
        a = analyse_pass(m, e["id"])
        if a["missed_better_option"]:
            best = next(o for o in a["options"] if o["better_option"])
            out.append({"pass_id": e["id"], "time": a["time"], "team": a["team"], "passer": a["passer"],
                        "pass_played_to": a["pass_played_to"], "better_option": best["label"],
                        "gain_pct": best["gain_vs_pass_played_pct"], "xp_pct": best["xp_pct"]})
    out.sort(key=lambda r: -r["gain_pct"])
    return out[:limit]
