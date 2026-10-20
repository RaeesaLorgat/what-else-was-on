"""Synthetic football match generator.

Simulates 22 players and the ball at 10 frames per second on a 105 x 68 m
pitch, and logs events (passes, receptions, interceptions, tackles,
pressures, shots, goals, ball out of play). Everything is made up: team
names, player names and every movement come from the random seed.

Coordinates: x runs 0 -> 105 (left goal to right goal), y runs 0 -> 68.
Team 0 attacks towards x = 105 and team 1 towards x = 0 for the whole
match (teams do not swap ends at half time, to keep things simple).

The model is deliberately simple but football-shaped:
- each team holds a formation that shifts with the ball;
- the defending team presses the ball carrier with its nearest player and
  covers with the next nearest;
- attackers make runs in behind;
- the ball carrier dribbles, then shoots or passes; passes are chosen
  mostly among teammates the carrier can see, and players are a bit
  cautious, so better options are sometimes skipped (which is what the
  agents will look for later);
- passes travel along the ground and can be intercepted by a defender who
  gets to the ball's path first; close pressure can win the ball.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

PITCH_X, PITCH_Y = 105.0, 68.0
HZ = 10
DT = 1.0 / HZ
GOAL_Y = (30.34, 37.66)          # goal mouth, 7.32 m wide, centred
VISION_HALF_ANGLE = math.radians(60)  # 120 degree cone

# formations as (x, y) slots for a team attacking towards x = 105, as a
# fraction of the pitch; index 0 is the goalkeeper
FORMATIONS = {
    "4-3-3": [(0.04, 0.50), (0.22, 0.15), (0.20, 0.38), (0.20, 0.62), (0.22, 0.85),
              (0.38, 0.30), (0.34, 0.50), (0.38, 0.70), (0.55, 0.15), (0.58, 0.50), (0.55, 0.85)],
    "4-4-2": [(0.04, 0.50), (0.22, 0.15), (0.20, 0.38), (0.20, 0.62), (0.22, 0.85),
              (0.38, 0.15), (0.36, 0.40), (0.36, 0.60), (0.38, 0.85), (0.55, 0.40), (0.55, 0.60)],
}
ROLES = {
    "4-3-3": ["GK", "RB", "CB", "CB", "LB", "CM", "DM", "CM", "RW", "ST", "LW"],
    "4-4-2": ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST", "ST"],
}
FIRST = ["Alex", "Sam", "Jordan", "Theo", "Kai", "Luca", "Noah", "Eli", "Milan", "Ravi", "Tomas",
         "Idris", "Jonah", "Mateo", "Ben", "Omar", "Leo", "Finn", "Ezra", "Nico", "Zane", "Ari"]
LAST = ["Hale", "Okafor", "Brandt", "Moreau", "Silva", "Novak", "Reyes", "Ward", "Kowal", "Ndlovu",
        "Fischer", "Costa", "Bjorn", "Mensah", "Varga", "Duarte", "Lindqvist", "Adeyemi", "Russo",
        "Petrov", "Haddad", "Quinn"]
TEAMS = [("Northbank FC", "4-3-3"), ("Riverside Athletic", "4-4-2")]


def threat(x: float, y: float, attack_right: bool) -> float:
    """A smooth stand-in for expected threat (xT): how dangerous it is to have
    the ball at (x, y). Rises towards the opponent's goal, highest centrally."""
    gx = PITCH_X if attack_right else 0.0
    d = math.hypot(gx - x, PITCH_Y / 2 - y)
    return 0.002 + 0.35 * math.exp(-d / 11.0)


@dataclass
class Player:
    pid: int
    team: int
    jersey: int
    name: str
    role: str
    top_speed: float


@dataclass
class Event:
    id: int
    t: float            # seconds from kick-off
    frame: int
    type: str           # pass, reception, interception, tackle, foul, pressure, shot, goal, out
    team: int
    player: int
    x: float
    y: float
    end_x: float | None = None
    end_y: float | None = None
    target: int | None = None     # intended receiver (passes)
    outcome: str | None = None    # complete / intercepted / out / stopped (passes); goal / saved / missed (shots)
    extra: dict = field(default_factory=dict)


class MatchSim:
    """One synthetic match. Call run() and then save()."""

    def __init__(self, seed: int = 1, minutes: float = 90.0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.minutes = minutes
        self.n_frames = int(minutes * 60 * HZ)
        self.players: list[Player] = []
        names = self.rng.permutation(len(FIRST)), self.rng.permutation(len(LAST))
        k = 0
        for team, (_, form) in enumerate(TEAMS):
            jerseys = [1, 2, 5, 4, 3, 8, 6, 10, 7, 9, 11]
            for i, role in enumerate(ROLES[form]):
                speed = 7.0 if role == "GK" else float(self.rng.uniform(7.6, 9.0))
                self.players.append(Player(k, team, jerseys[i], f"{FIRST[names[0][k]]} {LAST[names[1][k]]}", role, speed))
                k += 1
        self.slots = np.array([self._slot(p) for p in self.players])          # (22, 2) base positions
        self.pos = self.slots.copy() + self.rng.normal(0, 0.5, (22, 2))
        self.vel = np.zeros((22, 2))
        self.ball = np.array([PITCH_X / 2, PITCH_Y / 2])
        self.ball_vel = np.zeros(2)
        self.tracks = np.zeros((self.n_frames, 22, 2), dtype=np.float32)
        self.ball_track = np.zeros((self.n_frames, 2), dtype=np.float32)
        self.possession_track = np.zeros(self.n_frames, dtype=np.int8)
        self.events: list[Event] = []
        self.score = [0, 0]
        # state machine
        self.team_in_poss = 0
        self.carrier: int | None = self._kickoff_taker(0)
        self.state = "carry"                 # carry | pass | shot | dead
        self.hold = 1.0                      # seconds before the carrier acts
        self.flight: dict | None = None
        self.dead_timer = 0.0
        self.pressing: dict[int, float] = {} # defender -> seconds in close contact
        self.runs: dict[int, tuple[np.ndarray, float]] = {}  # attacker -> (target, seconds left)
        self.last_seen = np.full((22, 22), -99.0)            # last time player i had j in view

    # ---------- setup helpers ----------
    def _slot(self, p: Player) -> np.ndarray:
        form = TEAMS[p.team][1]
        fx, fy = FORMATIONS[form][self._role_index(p)]
        x, y = fx * PITCH_X, fy * PITCH_Y
        if p.team == 1:                      # team 1 attacks to the left: mirror
            x, y = PITCH_X - x, PITCH_Y - y
        return np.array([x, y])

    def _role_index(self, p: Player) -> int:
        return p.pid - 11 * p.team

    def _kickoff_taker(self, team: int) -> int:
        idx = [p.pid for p in self.players if p.team == team and p.role == "ST"]
        return idx[0]

    def team_ids(self, team: int) -> np.ndarray:
        return np.arange(11) + 11 * team

    def attack_right(self, team: int) -> bool:
        return team == 0

    # ---------- main loop ----------
    def run(self) -> "MatchSim":
        for f in range(self.n_frames):
            t = f * DT
            if f == self.n_frames // 2:
                self._restart_kickoff(1, f)          # second half: other team kicks off
            self._update_ball(f, t)
            self._move_players(f, t)
            self._update_vision(t)
            self.tracks[f] = self.pos
            self.ball_track[f] = self.ball
            self.possession_track[f] = self.team_in_poss
        return self

    # ---------- players ----------
    def _targets(self, t: float) -> np.ndarray:
        """Where every player wants to be this frame."""
        tgt = np.zeros((22, 2))
        bx, by = self.ball
        att = self.team_in_poss
        for p in self.players:
            sx, sy = self.slots[p.pid]
            own_goal_x = 0.0 if self.attack_right(p.team) else PITCH_X
            if p.role == "GK":
                gy = np.clip(by, GOAL_Y[0] - 2, GOAL_Y[1] + 2)
                tgt[p.pid] = [own_goal_x + (3 if p.team == 0 else -3), gy]
                continue
            # the whole block slides with the ball; the attacking team pushes up
            push = 12.0 if p.team == att else -4.0
            push = push if self.attack_right(p.team) else -push
            x = 0.55 * sx + 0.45 * bx + push
            y = 0.70 * sy + 0.30 * by
            tgt[p.pid] = [x, y]
        # attacking runs in behind
        for pid, (rt, left) in list(self.runs.items()):
            if left <= 0 or self.players[pid].team != att:
                del self.runs[pid]
                continue
            tgt[pid] = rt
            self.runs[pid] = (rt, left - DT)
        if self.state == "carry" and self.carrier is not None and self.rng.random() < 0.02:
            self._start_run(att)
        # defending team: nearest presses the ball, second nearest covers, others mark loosely
        dfd = 1 - att
        d_ids = [i for i in self.team_ids(dfd) if self.players[i].role != "GK"]
        focus = self.ball if self.state != "pass" else self.flight["aim"]
        dist = {i: float(np.hypot(*(self.pos[i] - focus))) for i in d_ids}
        order = sorted(d_ids, key=dist.get)
        if self.state in ("carry", "pass") and order:
            tgt[order[0]] = focus
            if len(order) > 1:
                own_goal = np.array([0.0 if self.attack_right(dfd) else PITCH_X, PITCH_Y / 2])
                tgt[order[1]] = focus + 0.35 * (own_goal - focus)
        a_ids = [i for i in self.team_ids(att) if self.players[i].role != "GK"]
        for i in order[2:]:
            near = min(a_ids, key=lambda j: np.hypot(*(self.pos[j] - self.pos[i])))
            tgt[i] = 0.6 * tgt[i] + 0.4 * self.pos[near]
        # in-flight pass: the receiver runs onto the ball
        if self.state == "pass":
            tgt[self.flight["target"]] = self.flight["aim"]
        if self.state == "carry" and self.carrier is not None:
            tgt[self.carrier] = self.pos[self.carrier] + self._dribble_dir(self.carrier) * 4.0
        return tgt

    def _start_run(self, team: int) -> None:
        cands = [p.pid for p in self.players if p.team == team and p.role in ("ST", "RW", "LW", "RM", "LM", "CM")
                 and p.pid != self.carrier and p.pid not in self.runs]
        if not cands:
            return
        pid = int(self.rng.choice(cands))
        dirx = 1 if self.attack_right(team) else -1
        rt = self.pos[pid] + np.array([dirx * self.rng.uniform(10, 20), self.rng.uniform(-8, 8)])
        rt = np.clip(rt, [2, 2], [PITCH_X - 2, PITCH_Y - 2])
        self.runs[pid] = (rt, self.rng.uniform(2.0, 4.0))

    def _move_players(self, f: int, t: float) -> None:
        tgt = self._targets(t)
        for p in self.players:
            i = p.pid
            if self.state == "carry" and i == self.carrier:
                cap = 5.5                                    # dribbling speed
            elif self.state == "pass" and i == self.flight["target"]:
                cap = p.top_speed
            elif i in self.runs:
                cap = p.top_speed
            else:
                cap = min(p.top_speed, 6.5)
            d = tgt[i] - self.pos[i]
            dist = float(np.hypot(*d))
            desired = d / dist * min(cap, dist * 1.2) if dist > 0.05 else np.zeros(2)
            dv = desired - self.vel[i]
            acc = float(np.hypot(*dv))
            max_dv = 4.5 * DT                               # max acceleration 4.5 m/s^2
            if acc > max_dv:
                dv = dv / acc * max_dv
            self.vel[i] = self.vel[i] + dv
            self.pos[i] = np.clip(self.pos[i] + self.vel[i] * DT, [0.5, 0.5], [PITCH_X - 0.5, PITCH_Y - 0.5])
        if self.state == "carry" and self.carrier is not None:
            self.ball = self.pos[self.carrier] + self._dribble_dir(self.carrier) * 0.6

    def _dribble_dir(self, i: int) -> np.ndarray:
        goal = np.array([PITCH_X if self.attack_right(self.players[i].team) else 0.0, PITCH_Y / 2])
        d = goal - self.pos[i]
        return d / max(float(np.hypot(*d)), 1e-6)

    def facing(self, i: int) -> np.ndarray:
        """Where player i is looking: the way he moves, or towards the opponents' goal when standing."""
        v = self.vel[i]
        sp = float(np.hypot(*v))
        if sp > 0.8:
            return v / sp
        return self._dribble_dir(i)

    def in_view(self, i: int, point: np.ndarray) -> bool:
        d = point - self.pos[i]
        n = float(np.hypot(*d))
        if n < 1e-6:
            return True
        cosang = float(np.dot(d / n, self.facing(i)))
        return cosang >= math.cos(VISION_HALF_ANGLE)

    def _update_vision(self, t: float) -> None:
        if self.state == "carry" and self.carrier is not None:
            i = self.carrier
            for j in self.team_ids(self.players[i].team):
                if j != i and self.in_view(i, self.pos[j]):
                    self.last_seen[i, j] = t

    # ---------- ball ----------
    def _update_ball(self, f: int, t: float) -> None:
        if self.state == "dead":
            self.dead_timer -= DT
            if self.dead_timer <= 0:
                self.state = "carry"
                self.hold = self.rng.uniform(0.5, 1.5)
            return
        if self.state == "carry":
            self._carry(f, t)
        elif self.state in ("pass", "shot"):
            self._flight(f, t)

    def _carry(self, f: int, t: float) -> None:
        i = self.carrier
        team = self.players[i].team
        # pressure and tackles
        for j in self.team_ids(1 - team):
            d = float(np.hypot(*(self.pos[j] - self.pos[i])))
            if d < 3.0 and j not in self.pressing:
                self.pressing[j] = 0.0
                self._log(f, "pressure", 1 - team, j, *self.pos[j], extra={"on": int(i)})
            if j in self.pressing:
                if d >= 3.5:
                    del self.pressing[j]
                    continue
                if d < 1.4:
                    self.pressing[j] += DT
                    if self.pressing[j] > 0.4 and self.rng.random() < 0.9 * DT:
                        if self.rng.random() < 0.4:          # a foul: free kick to the carrier's team
                            self._log(f, "foul", 1 - team, j, *self.pos[i], extra={"on": int(i)})
                            self._restart(f, i, delay=float(self.rng.uniform(12, 25)))
                        else:
                            self._log(f, "tackle", 1 - team, j, *self.pos[i], extra={"won_from": int(i)})
                            self._change_possession(j)
                        return
        self.hold -= DT
        under_pressure = any(np.hypot(*(self.pos[j] - self.pos[i])) < 3.0 for j in self.team_ids(1 - team))
        if self.hold > 0 and not (under_pressure and self.rng.random() < 0.8 * DT):
            return
        # shoot?
        gx = PITCH_X if self.attack_right(team) else 0.0
        dgoal = math.hypot(gx - self.pos[i][0], PITCH_Y / 2 - self.pos[i][1])
        if dgoal < 24 and self.rng.random() < (0.35 if dgoal < 16 else 0.08):
            self._shoot(f, t, i, dgoal)
            return
        self._pass(f, t, i)

    def _arrival_margin(self, start: np.ndarray, aim: np.ndarray, speed: float, team: int) -> float:
        """Seconds to spare before the quickest opponent could reach the ball's path (negative = cut out)."""
        L = float(np.hypot(*(aim - start)))
        best = 99.0
        opp = self.team_ids(1 - team)
        for s in np.arange(2.0, L + 0.01, 1.0):
            pt = start + (aim - start) * s / max(L, 1e-6)
            tb = s / speed
            q = self.pos[opp] + self.vel[opp] * 0.7
            to = 0.7 + np.hypot(*(pt - q).T) / 7.0
            best = min(best, float(np.min(to)) - tb)
        return best

    def _pass(self, f: int, t: float, i: int) -> None:
        team = self.players[i].team
        here = self.threat_at(self.pos[i], team)
        options = []
        for j in self.team_ids(team):
            if j == i:
                continue
            d = float(np.hypot(*(self.pos[j] - self.pos[i])))
            if d < 4 or d > 45:
                continue
            speed = float(np.clip(8 + 0.25 * d, 9, 17))
            aim = self.pos[j] + self.vel[j] * (d / speed)
            aim = np.clip(aim, [1, 1], [PITCH_X - 1, PITCH_Y - 1])
            margin = self._arrival_margin(self.pos[i], aim, speed, team)
            gain = self.threat_at(aim, team) - here
            seen = self.in_view(i, self.pos[j]) or (t - self.last_seen[i, j] < 2.0)
            options.append((j, aim, speed, margin, gain, seen))
        if not options:
            self.hold = 0.5
            return
        # players are cautious: safety counts for more than threat, and they
        # rarely pass to someone they have not seen; this leaves better options unused
        score = np.array([3.5 * np.tanh(m / 0.6) + 25.0 * g + (0.0 if s else -5.0) for (_, _, _, m, g, s) in options])
        prob = np.exp(score - score.max())
        prob /= prob.sum()
        k = int(self.rng.choice(len(options), p=prob))
        j, aim, speed, margin, gain, seen = options[k]
        aim = aim + self.rng.normal(0, 0.5, 2)                 # passes are not perfect
        self.flight = {"from": self.pos[i].copy(), "aim": aim, "speed": speed, "target": int(j),
                       "passer": int(i), "event": len(self.events)}
        self._log(f, "pass", team, i, *self.pos[i], end_x=float(aim[0]), end_y=float(aim[1]), target=int(j),
                  extra={"in_view": bool(self.in_view(i, self.pos[j])), "speed": round(speed, 1)})
        d = aim - self.pos[i]
        self.ball_vel = d / max(float(np.hypot(*d)), 1e-6) * speed
        self.state = "pass"
        self.carrier = None
        self.pressing.clear()

    def threat_at(self, p: np.ndarray, team: int) -> float:
        return threat(float(p[0]), float(p[1]), self.attack_right(team))

    def _shoot(self, f: int, t: float, i: int, dgoal: float) -> None:
        team = self.players[i].team
        gx = PITCH_X if self.attack_right(team) else 0.0
        aim = np.array([gx, self.rng.uniform(GOAL_Y[0] - 3, GOAL_Y[1] + 3)])
        xg = 1 / (1 + math.exp(0.18 * dgoal - 0.3))         # simple distance-based chance
        self.flight = {"from": self.pos[i].copy(), "aim": aim, "speed": 25.0, "target": None,
                       "passer": int(i), "event": len(self.events), "xg": xg}
        self._log(f, "shot", team, i, *self.pos[i], end_x=float(aim[0]), end_y=float(aim[1]),
                  extra={"xg": round(xg, 3)})
        d = aim - self.pos[i]
        self.ball_vel = d / max(float(np.hypot(*d)), 1e-6) * 25.0
        self.state = "shot"
        self.carrier = None

    def _flight(self, f: int, t: float) -> None:
        fl = self.flight
        team = self.players[fl["passer"]].team
        self.ball = self.ball + self.ball_vel * DT
        ev = self.events[fl["event"]]
        if self.state == "shot":
            gx = PITCH_X if self.attack_right(team) else 0.0
            if (self.ball[0] >= gx - 1.0) if self.attack_right(team) else (self.ball[0] <= gx + 1.0):
                on_target = GOAL_Y[0] <= self.ball[1] <= GOAL_Y[1]
                if on_target and self.rng.random() < min(1.0, fl["xg"] / 0.55):
                    ev.outcome = "goal"
                    self.score[team] += 1
                    self._log(f, "goal", team, fl["passer"], *self.ball, extra={"score": list(self.score)})
                    self._restart_kickoff(1 - team, f)
                else:
                    ev.outcome = "saved" if on_target else "missed"
                    gk = [p.pid for p in self.players if p.team == 1 - team and p.role == "GK"][0]
                    self._restart(f, gk, delay=float(self.rng.uniform(5, 9)))
            return
        # interception: an opponent close to the ball wins it
        opp = self.team_ids(1 - team)
        dopp = np.hypot(*(self.pos[opp] - self.ball).T)
        if dopp.min() < 1.0 and float(np.hypot(*(self.ball - fl["from"]))) > 2.0:
            j = int(opp[int(np.argmin(dopp))])
            ev.outcome = "intercepted"
            self._log(f, "interception", 1 - team, j, *self.ball, extra={"pass_event": fl["event"]})
            self._change_possession(j)
            return
        # reception by any teammate close to the ball (usually the target)
        mates = [j for j in self.team_ids(team) if j != fl["passer"]]
        dm = np.hypot(*(self.pos[mates] - self.ball).T)
        if dm.min() < 1.8:
            j = int(mates[int(np.argmin(dm))])
            ev.outcome = "complete"
            ev.extra["receiver"] = j
            self._log(f, "reception", team, j, *self.ball, extra={"pass_event": fl["event"]})
            self.carrier = j
            self.state = "carry"
            self.hold = float(self.rng.uniform(2.2, 5.0))
            self.flight = None
            return
        # out of play, or the pass has gone well past everyone
        travelled = float(np.hypot(*(self.ball - fl["from"])))
        planned = float(np.hypot(*(fl["aim"] - fl["from"])))
        if not (0 <= self.ball[0] <= PITCH_X and 0 <= self.ball[1] <= PITCH_Y) or travelled > planned + 8:
            ev.outcome = "out"
            self._log(f, "out", team, fl["passer"], *np.clip(self.ball, 0, [PITCH_X, PITCH_Y]))
            nearest = int(opp[int(np.argmin(np.hypot(*(self.pos[opp] - self.ball).T)))])
            self.ball = np.clip(self.ball, [0.5, 0.5], [PITCH_X - 0.5, PITCH_Y - 0.5])
            self._restart(f, nearest, delay=float(self.rng.uniform(6, 12)))

    # ---------- possession changes ----------
    def _change_possession(self, j: int) -> None:
        self.team_in_poss = self.players[j].team
        self.carrier = j
        self.state = "carry"
        self.hold = float(self.rng.uniform(0.3, 1.2))
        self.flight = None
        self.pressing.clear()
        self.runs.clear()

    def _restart(self, f: int, j: int, delay: float) -> None:
        self._change_possession(j)
        self.ball = self.pos[j].copy()
        self.state = "dead"
        self.dead_timer = delay

    def _restart_kickoff(self, team: int, f: int) -> None:
        if self.flight is not None and self.events[self.flight["event"]].outcome is None:
            self.events[self.flight["event"]].outcome = "stopped"     # half time blew while the ball was moving
        self.pos = self.slots.copy()
        self.vel[:] = 0
        self.ball = np.array([PITCH_X / 2, PITCH_Y / 2])
        j = self._kickoff_taker(team)
        self.pos[j] = self.ball - np.array([0.8 if team == 0 else -0.8, 0])
        self._restart(f, j, delay=8.0)

    def _log(self, f: int, typ: str, team: int, player: int, x: float, y: float, **kw) -> None:
        self.events.append(Event(len(self.events), round(f * DT, 1), f, typ, int(team), int(player),
                                 round(float(x), 2), round(float(y), 2), **kw))

    # ---------- output ----------
    def summary(self) -> dict:
        passes = [e for e in self.events if e.type == "pass"]
        done = [e for e in passes if e.outcome == "complete"]
        return {
            "score": self.score,
            "passes": len(passes),
            "pass_completion": round(len(done) / max(len(passes), 1), 3),
            "blind_passes": sum(1 for e in passes if not e.extra.get("in_view")),
            "shots": sum(1 for e in self.events if e.type == "shot"),
            "tackles": sum(1 for e in self.events if e.type == "tackle"),
            "interceptions": sum(1 for e in self.events if e.type == "interception"),
            "possession_team0": round(float(np.mean(self.possession_track == 0)), 3),
        }

    def save(self, folder: str | Path = "data") -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"match_{self.seed}"
        np.savez_compressed(f"{stem}_tracking.npz", players=self.tracks, ball=self.ball_track,
                            possession=self.possession_track, hz=HZ)
        meta = {
            "seed": self.seed, "minutes": self.minutes, "hz": HZ, "pitch": [PITCH_X, PITCH_Y],
            "synthetic": True,
            "teams": [{"team": k, "name": n, "formation": fm, "attacks": "right" if k == 0 else "left"}
                      for k, (n, fm) in enumerate(TEAMS)],
            "players": [asdict(p) for p in self.players],
            "summary": self.summary(),
            "events": [asdict(e) for e in self.events],
        }
        with open(f"{stem}_events.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        return Path(f"{stem}_events.json")


def load_match(seed: int, folder: str | Path = "data") -> tuple[dict, dict]:
    """Read a saved match back: (meta with events, tracking arrays)."""
    stem = Path(folder) / f"match_{seed}"
    with open(f"{stem}_events.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    z = np.load(f"{stem}_tracking.npz")
    return meta, {k: z[k] for k in z.files}
