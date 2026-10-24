"""The replay app: a small web backend (FastAPI) and the page in web/.

    python scripts/serve.py            # then open http://localhost:8000

API
    GET  /api/matches                              the generated matches
    GET  /api/matches/{seed}                       teams, players, summary and the key moments
    GET  /api/matches/{seed}/moments/{pass_id}     a 9 second clip around the pass, the tool analysis,
                                                   and the agents' story if it has been run
    POST /api/matches/{seed}/moments/{pass_id}/run run the agent team live on this pass (needs .env)

Stories made by scripts/run_pipeline.py (data/stories_<seed>.json) are shown straight away;
running the agents from the page adds to that file.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .checks import CLOSE_CALL_S
from .tools import (BALL_SPEED, CONE_HALF_ANGLE, REACT, VMAX, analyse_pass, arrival, facing_direction,
                    find_missed_passes, get_match, lead_aim)
from .xt import model_name

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("WHATELSE_DATA", ROOT / "data"))
WEB = ROOT / "web"
BEFORE_S, AFTER_S = 5.0, 4.0           # clip length around the pass
TOP_MOMENTS = 10

app = FastAPI(title="What Else Was On", description="Replay of synthetic matches with the agents' verdicts")
_locks: dict[int, asyncio.Lock] = {}


def _seeds() -> list[int]:
    out = []
    for f in DATA.glob("match_*_events.json"):
        try:
            seed = int(f.stem.split("_")[1])
        except ValueError:
            continue
        if (DATA / f"match_{seed}_tracking.npz").exists():
            out.append(seed)
    return sorted(out)


def _match(seed: int):
    if seed not in _seeds():
        raise HTTPException(404, f"match {seed} has not been generated (python scripts/generate_match.py --seed {seed})")
    return get_match(seed, DATA)


def _stories(seed: int) -> dict:
    f = DATA / f"stories_{seed}.json"
    if not f.exists():
        return {"seed": seed, "stories": []}
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return {"seed": seed, "stories": []}


def _story_for(seed: int, pass_id: int) -> dict | None:
    return next((s for s in _stories(seed).get("stories", []) if s.get("pass_id") == pass_id), None)


@app.get("/api/matches")
def matches() -> list[dict]:
    out = []
    for seed in _seeds():
        meta = json.loads((DATA / f"match_{seed}_events.json").read_text())
        out.append({"seed": seed, "teams": [t["name"] for t in meta["teams"]],
                    "score": meta["summary"]["score"], "stories": len(_stories(seed).get("stories", []))})
    return out


@app.get("/api/matches/{seed}")
def match(seed: int) -> dict:
    m = _match(seed)
    stories = {s["pass_id"]: s for s in _stories(seed).get("stories", [])}
    moments = []
    for r in find_missed_passes(m, limit=TOP_MOMENTS):
        s = stories.get(r["pass_id"])
        moments.append({**r, "status": s["status"] if s else None,
                        "commentary": s["commentary"] if s else None})
    seen = {mo["pass_id"] for mo in moments}
    for pid, s in stories.items():           # stories for passes outside the top list
        if pid not in seen:
            v = s["verdict"]
            moments.append({"pass_id": pid, "time": v.get("time"), "passer": v.get("passer"),
                            "pass_played_to": v.get("pass_played_to"), "better_option": v.get("better_option"),
                            "gain_pct": (v.get("evidence") or {}).get("gain_pct"),
                            "xp_pct": (v.get("evidence") or {}).get("better_option_xp_pct"),
                            "status": s["status"], "commentary": s["commentary"]})
    return {"seed": seed, "teams": m.meta["teams"], "players": m.players, "summary": m.meta["summary"],
            "hz": m.meta.get("hz", 10), "pitch": m.meta.get("pitch", [105, 68]),
            "threat_model": model_name(), "moments": moments,
            "recap": _stories(seed).get("recap")}


def _r(a: np.ndarray) -> list:
    return np.round(a, 2).tolist()


@app.get("/api/matches/{seed}/moments/{pass_id}")
def moment(seed: int, pass_id: int) -> dict:
    m = _match(seed)
    try:
        a = analyse_pass(m, pass_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    e = m.event(pass_id)
    hz = m.meta.get("hz", 10)
    f0 = max(0, e["frame"] - int(BEFORE_S * hz))
    f1 = min(len(m.tracks) - 1, e["frame"] + int(AFTER_S * hz))
    i = e["player"]
    face = facing_direction(m, i, e["frame"])

    options = []
    for o in a["options"]:
        arr = arrival(m, i, o["player"], e["frame"])
        options.append({**o, "aim": _r(lead_aim(m, i, o["player"], e["frame"])),
                        "closest_defender_id": arr.get("closest_defender_id"),
                        "closest_point": arr.get("closest_point")})
    return {
        "pass_id": pass_id, "seed": seed,
        "clip": {"start_frame": f0, "pass_index": e["frame"] - f0, "hz": hz,
                 "players": _r(m.tracks[f0:f1 + 1]), "ball": _r(m.ball[f0:f1 + 1])},
        "pass": {"passer": i, "target": e["target"], "outcome": e["outcome"],
                 "from": [e["x"], e["y"]], "to": [e["end_x"], e["end_y"]],
                 "velocities": _r(m.vel(e["frame"]))},
        "cone": {"facing": _r(face), "half_angle_deg": CONE_HALF_ANGLE},
        "model": {"ball_speed": BALL_SPEED, "react_s": REACT, "vmax": VMAX, "tight_below_s": CLOSE_CALL_S,
                  "threat_model": model_name()},
        "analysis": {**a, "options": options},
        "story": _story_for(seed, pass_id),
    }


@app.post("/api/matches/{seed}/moments/{pass_id}/run")
async def run_agents(seed: int, pass_id: int) -> dict:
    m = _match(seed)
    try:
        m.event(pass_id)
        from .llm import load_settings
        load_settings()
    except ValueError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(503, f"The agents need a Foundry model: {e}")
    from .pipeline import run_pass

    lock = _locks.setdefault(seed, asyncio.Lock())
    story = await run_pass(m, pass_id)
    async with lock:
        data = _stories(seed)
        data["stories"] = [s for s in data.get("stories", []) if s.get("pass_id") != pass_id] + [story]
        (DATA / f"stories_{seed}.json").write_text(json.dumps(data, indent=2))
    return story


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/healthz")
def health() -> dict:
    return {"ok": True, "matches": _seeds()}


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")
