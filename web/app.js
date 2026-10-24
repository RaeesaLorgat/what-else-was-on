// What Else Was On: replay page.
// Plays a 9 second clip around a pass, stops at the moment of the pass, draws the passer's
// vision cone, the pass played and the option that was on, and can play that option as a
// "ghost" pass with the quickest defender chasing it (same reach model as the tools).

const $ = (id) => document.getElementById(id);
const PITCH_X = 105, PITCH_Y = 68, MARGIN = 3;
const css = getComputedStyle(document.documentElement);
const C = (name) => css.getPropertyValue(name).trim();

const state = {
  seed: null, match: null, moment: null,
  t: 0, playing: false, speed: 1, last: null,
  mode: "clip",            // "clip" or "ghost"
  stoppedAtPass: false, ghostT: 0, ghostDone: false,
  follow: true, cam: null,  // camera: follows the action (zoomed) or shows the whole pitch
};
const FOLLOW_W = 64;        // metres of pitch shown across when following the action

// ---------- data ----------
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

async function init() {
  const list = await api("/api/matches");
  const sel = $("matchSelect");
  sel.innerHTML = "";
  if (!list.length) {
    $("storyBox").innerHTML = `<p class="empty">No matches yet. Run <code>python scripts/generate_match.py --seed 1</code>.</p>`;
    return;
  }
  for (const mt of list) {
    const o = document.createElement("option");
    o.value = mt.seed;
    o.textContent = `Match ${mt.seed}: ${mt.teams[0]} ${mt.score[0]}-${mt.score[1]} ${mt.teams[1]}`;
    sel.appendChild(o);
  }
  sel.onchange = () => loadMatch(+sel.value);
  await loadMatch(list[0].seed);
}

async function loadMatch(seed) {
  state.seed = seed;
  const m = await api(`/api/matches/${seed}`);
  state.match = m;
  const [a, b] = m.teams;
  $("scoreboard").innerHTML =
    `<span class="t0">${esc(a.name)}</span><span class="score">${m.summary.score[0]} - ${m.summary.score[1]}</span>` +
    `<span class="t1">${esc(b.name)}</span>`;
  $("legendT0").textContent = a.name;
  $("legendT1").textContent = b.name;
  $("threatModel").textContent = m.threat_model;
  renderMoments();
  if (m.recap) { $("recapBox").hidden = false; $("recapText").textContent = m.recap; }
  else $("recapBox").hidden = true;
  if (m.moments.length) await loadMoment(m.moments[0].pass_id);
}

function renderMoments() {
  const ol = $("momentList");
  ol.innerHTML = "";
  for (const mo of state.match.moments) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.dataset.pid = mo.pass_id;
    const badge = mo.status ? `<span class="badge ${mo.status}">${mo.status === "approved" ? "agents approved" : "plain verdict"}</span>`
                            : `<span class="badge">not run</span>`;
    b.innerHTML =
      `<div class="m-top"><span class="m-time">${esc(mo.time)}</span>${badge}</div>` +
      `<div class="m-who">${esc(short(mo.passer))} to ${esc(short(mo.pass_played_to))}</div>` +
      `<div class="m-what"><b>${esc(short(mo.better_option))}</b> was on, +${mo.gain_pct} pts threat, ${mo.xp_pct}% to complete</div>`;
    b.onclick = () => loadMoment(mo.pass_id);
    li.appendChild(b);
    ol.appendChild(li);
  }
}

async function loadMoment(pid) {
  for (const b of document.querySelectorAll("#momentList button"))
    b.setAttribute("aria-current", +b.dataset.pid === pid ? "true" : "false");
  const mo = await api(`/api/matches/${state.seed}/moments/${pid}`);
  state.moment = mo;
  state.t = 0; state.mode = "clip"; state.stoppedAtPass = false; state.ghostT = 0; state.ghostDone = false;
  const clipLen = (mo.clip.players.length - 1) / mo.clip.hz;
  $("scrub").max = clipLen;
  const a = mo.analysis;
  $("momentHead").innerHTML =
    `<span class="t">${esc(a.time)}</span><span class="who">${esc(a.passer)} to ${esc(a.pass_played_to)}</span>` +
    `<span class="m-what">${esc(a.team)}, pass ${a.pass_id}</span>`;
  $("overlayNote").hidden = true;
  $("ghostBtn").disabled = true;
  $("continueBtn").disabled = true;
  renderStory();
  setPlaying(true);
}

// ---------- story panel ----------
function renderStory(runMsg) {
  const mo = state.moment, a = mo.analysis, s = mo.story;
  const best = bestOption();
  const played = a.options.find((o) => o.was_pass_played);
  let html = `<h2><span>The verdict</span>${s ? `<span class="badge ${s.status}">${s.status === "approved" ? "agents approved" : "plain verdict from the tools"}</span>` : `<span class="badge">agents not run yet</span>`}</h2>`;
  if (s) {
    html += `<p class="commentary">${esc(s.commentary)}</p>`;
    if (s.pundit) html += `<p class="pundit">${esc(s.pundit)}</p>`;
  } else {
    html += `<p class="empty">The tools have analysed this pass. Run the agents to get the commentary.</p>`;
  }
  if (best) {
    const tight = best.closest_call_s < mo.model.tight_below_s;
    html += `<div class="tiles">
      <div class="tile"><div class="k">Threat where it went</div><div class="v">${a.threat_of_pass_played_pct}%</div></div>
      <div class="tile good"><div class="k">Threat at ${esc(short(best.label))}</div><div class="v">${best.threat_pct}% <small>+${best.gain_vs_pass_played_pct} pts</small></div></div>
      <div class="tile"><div class="k">Chance that pass completes</div><div class="v">${best.xp_pct}%</div></div>
      <div class="tile ${tight ? "warn" : ""}"><div class="k">Beats quickest defender by</div><div class="v">${best.closest_call_s} s <small>${tight ? "tight" : "comfortable"}</small></div></div>
    </div>`;
  }
  if (s) {
    html += `<div class="section-title">How the agents got there</div><ol class="trail">`;
    for (const step of s.trail) html += trailItem(step);
    html += `</ol>`;
    const v = s.verdict;
    if (v && v.explanation) html += `<div class="section-title">Analyst's verdict</div><p class="pundit">${esc(v.explanation)}</p>`;
  }
  html += `<div class="run-row"><button class="btn primary" id="runBtn">${s ? "Run the agents again" : "Run the agents on this pass"}</button>
           <span class="run-msg" id="runMsg">${runMsg ? esc(runMsg) : ""}</span></div>`;
  html += `<details><summary>Every teammate at the moment of the pass</summary>${optionsTable(a.options)}
           <p class="run-msg">${esc(a.rule)}.</p></details>`;
  $("storyBox").innerHTML = html;
  $("runBtn").onclick = runAgents;
}

function trailItem(t) {
  if (t.agent === "Analyst") {
    const v = t.verdict || {};
    const what = v.verdict === "missed_better_option" ? `says ${esc(short(v.better_option))} was on (+${(v.evidence || {}).gain_pct} pts)`
               : v.verdict === "good_choice" ? "says it was a good choice" : "reply was not a verdict";
    return `<li class="info"><span class="who">Analyst</span>draft ${t.draft}: ${what}</li>`;
  }
  if (t.agent === "Checker") {
    const on = t.draft ? `draft ${t.draft}` : `${t.on === "Recap" ? "recap" : "commentary"}`;
    const probs = (t.problems || []).map((p) => `<li>${esc(p)}</li>`).join("");
    return `<li class="${t.approved ? "ok" : "no"}"><span class="who">Checker</span>${on}: ${t.approved ? "approved" : "rejected"}${probs ? `<ul>${probs}</ul>` : ""}</li>`;
  }
  if (t.agent === "Narrator") return `<li class="ok"><span class="who">Narrator</span>wrote the commentary</li>`;
  return `<li class="warn"><span class="who">${esc(t.agent)}</span>${esc(t.note || "")}</li>`;
}

function optionsTable(opts) {
  let h = `<table><thead><tr><th>Teammate</th><th>In view</th><th>Offside</th><th>Threat</th><th>vs played</th><th>Arrives</th><th>xP</th></tr></thead><tbody>`;
  for (const o of opts) {
    const cls = o.better_option ? "better" : o.was_pass_played ? "played" : (!o.in_view || o.offside) ? "dim" : "";
    h += `<tr class="${cls}"><td>${esc(o.label)}${o.was_pass_played ? " (played)" : ""}</td><td>${o.in_view ? "yes" : "no"}</td>
          <td>${o.offside ? "yes" : "no"}</td><td>${o.threat_pct}%</td><td>${o.gain_vs_pass_played_pct > 0 ? "+" : ""}${o.gain_vs_pass_played_pct}</td>
          <td>${o.arrives ? "yes" : "no"}</td><td>${o.xp_pct}%</td></tr>`;
  }
  return h + `</tbody></table>`;
}

async function runAgents() {
  const btn = $("runBtn");
  btn.disabled = true;
  $("runMsg").textContent = "Analyst, Checker and Narrator are working (about 20 to 60 s)...";
  try {
    const story = await api(`/api/matches/${state.seed}/moments/${state.moment.pass_id}/run`, { method: "POST" });
    state.moment.story = story;
    const mo = state.match.moments.find((x) => x.pass_id === story.pass_id);
    if (mo) { mo.status = story.status; mo.commentary = story.commentary; }
    renderMoments();
    for (const b of document.querySelectorAll("#momentList button"))
      b.setAttribute("aria-current", +b.dataset.pid === story.pass_id ? "true" : "false");
    renderStory("Done.");
  } catch (e) {
    renderStory(e.message);
  }
}

// ---------- geometry ----------
function bestOption() {
  return state.moment.analysis.options.find((o) => o.better_option) || null;
}
function frameAt(t) {
  const c = state.moment.clip, k = t * c.hz, i = Math.min(Math.floor(k), c.players.length - 1);
  const j = Math.min(i + 1, c.players.length - 1), f = k - i;
  const lerp = (a, b) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
  return { players: c.players[i].map((p, n) => lerp(p, c.players[j][n])), ball: lerp(c.ball[i], c.ball[j]), index: k };
}
const passT = () => state.moment.clip.pass_index / state.moment.clip.hz;

// Ghost pass: ball from the passer to the aim point at the model's ball speed; the quickest
// defender keeps his momentum for the reaction time, then sprints at top speed to the point
// on the ball's path where he comes closest (both as in tools.arrival).
function ghost(t) {
  const mo = state.moment, best = bestOption(), md = mo.model;
  const P = mo.clip.players[mo.clip.pass_index], from = P[mo.pass.passer], aim = best.aim;
  const L = Math.hypot(aim[0] - from[0], aim[1] - from[1]);
  const tb = Math.min(t, L / md.ball_speed);
  const ball = [from[0] + (aim[0] - from[0]) * tb * md.ball_speed / L, from[1] + (aim[1] - from[1]) * tb * md.ball_speed / L];
  let def = null;
  if (best.closest_defender_id != null && best.closest_point) {
    const d = best.closest_defender_id, p0 = P[d], v = mo.pass.velocities[d], pt = best.closest_point;
    const tr = Math.min(t, md.react_s);
    let q = [p0[0] + v[0] * tr, p0[1] + v[1] * tr];
    if (t > md.react_s) {
      const q0 = [p0[0] + v[0] * md.react_s, p0[1] + v[1] * md.react_s];
      const dist = Math.hypot(pt[0] - q0[0], pt[1] - q0[1]), run = Math.min(dist, (t - md.react_s) * md.vmax);
      q = dist > 0 ? [q0[0] + (pt[0] - q0[0]) * run / dist, q0[1] + (pt[1] - q0[1]) * run / dist] : q0;
    }
    def = { id: d, pos: q };
  }
  // the receiver keeps running at his current velocity (the pass is aimed where he will be)
  const r = best.player, rv = mo.pass.velocities[r], tr2 = Math.min(t, L / md.ball_speed);
  const recv = { id: r, pos: [P[r][0] + rv[0] * tr2, P[r][1] + rv[1] * tr2] };
  return { ball, def, recv, done: t >= L / md.ball_speed + 0.6, ballTime: L / md.ball_speed };
}

// ---------- drawing ----------
const canvas = $("pitch"), ctx = canvas.getContext("2d");
let S = 1, camX0 = -MARGIN, camY0 = -MARGIN;
const FULL_W = PITCH_X + 2 * MARGIN, FULL_H = PITCH_Y + 2 * MARGIN;
function resize() {
  const w = canvas.clientWidth, dpr = window.devicePixelRatio || 1;
  const h = w * FULL_H / FULL_W;
  canvas.style.height = h + "px";
  canvas.width = w * dpr; canvas.height = h * dpr;
}
// camera: centre (cx, cy) and width in metres, eased towards a target every frame
function setCamera(target) {
  const want = state.follow ? { ...target, w: FOLLOW_W } : { x: PITCH_X / 2, y: PITCH_Y / 2, w: FULL_W };
  if (!state.cam) state.cam = want;
  const k = 0.12;
  state.cam = { x: state.cam.x + (want.x - state.cam.x) * k, y: state.cam.y + (want.y - state.cam.y) * k,
                w: state.cam.w + (want.w - state.cam.w) * k };
  const w = state.cam.w, h = w * FULL_H / FULL_W;
  S = canvas.width / w;
  camX0 = Math.min(Math.max(state.cam.x - w / 2, -MARGIN), FULL_W - MARGIN - w);
  camY0 = Math.min(Math.max(state.cam.y - h / 2, -MARGIN), FULL_H - MARGIN - h);
}
const X = (x) => (x - camX0) * S, Y = (y) => (y - camY0) * S;

function drawPitch() {
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#123d24"; ctx.fillRect(0, 0, W, H);
  for (let k = 0; k < 14; k++) {
    ctx.fillStyle = k % 2 ? "#14432a" : "#163f28";
    ctx.fillRect(X(k * PITCH_X / 14), Y(-MARGIN), PITCH_X / 14 * S + 1, FULL_H * S);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = Math.max(1, 0.15 * S);
  ctx.strokeRect(X(0), Y(0), PITCH_X * S, PITCH_Y * S);
  line([PITCH_X / 2, 0], [PITCH_X / 2, PITCH_Y]);
  circle([PITCH_X / 2, PITCH_Y / 2], 9.15);
  for (const side of [0, 1]) {
    const x0 = side ? PITCH_X : 0, dir = side ? -1 : 1;
    ctx.strokeRect(X(Math.min(x0, x0 + dir * 16.5)), Y(PITCH_Y / 2 - 20.16), 16.5 * S, 40.32 * S);
    ctx.strokeRect(X(Math.min(x0, x0 + dir * 5.5)), Y(PITCH_Y / 2 - 9.16), 5.5 * S, 18.32 * S);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillRect(X(side ? PITCH_X : -1.2), Y(PITCH_Y / 2 - 3.66), 1.2 * S, 7.32 * S);
  }
}
function line(a, b) { ctx.beginPath(); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); ctx.stroke(); }
function circle(c, r) { ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), r * S, 0, 2 * Math.PI); ctx.stroke(); }

function arrow(a, b, color, dashed, width) {
  ctx.save();
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width * S;
  if (dashed) ctx.setLineDash([1.2 * S, 0.8 * S]);
  line(a, b);
  ctx.setLineDash([]);
  const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), h = 1.6;
  ctx.beginPath();
  ctx.moveTo(X(b[0]), Y(b[1]));
  ctx.lineTo(X(b[0] - h * Math.cos(ang - 0.4)), Y(b[1] - h * Math.sin(ang - 0.4)));
  ctx.lineTo(X(b[0] - h * Math.cos(ang + 0.4)), Y(b[1] - h * Math.sin(ang + 0.4)));
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

let labelQueue = [];   // labels are drawn last, on top of the players
// opts: below (put it under the point), away (a point to keep clear of: the label goes on the
// other side of p from it, so it does not cover the passer), row (stack a second label under the first)
function label(p, text, color, opts) { labelQueue.push([p, text, color, opts === true ? { below: true } : (opts || {})]); }
function drawLabel(p, text, color, o) {
  ctx.save();
  const dpr = window.devicePixelRatio || 1;
  ctx.font = `600 ${12 * dpr}px Inter, Segoe UI, sans-serif`;
  const w = ctx.measureText(text).width + 10 * dpr, h = 20 * dpr;
  const left = o.away ? o.away[0] > p[0] : false;
  const below = o.away ? o.away[1] < p[1] : !!o.below;
  let x = left ? X(p[0]) - 1.8 * S - w : X(p[0]) + 1.8 * S;
  let y = below ? Y(p[1]) + 1.8 * S : Y(p[1]) - 1.8 * S - h;
  y += (o.row || 0) * (h + 4 * dpr) * (below ? 1 : -1);
  x = Math.min(Math.max(x, 4), canvas.width - w - 4); y = Math.min(Math.max(y, 4), canvas.height - h - 4);
  ctx.fillStyle = "rgba(8,14,20,0.85)"; ctx.fillRect(x, y, w, h);
  ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, x + 5, y + h / 2);
  ctx.restore();
}

function drawCone(from) {
  const mo = state.moment, f = mo.cone.facing, half = mo.cone.half_angle_deg * Math.PI / 180, R = 38;
  const a0 = Math.atan2(f[1], f[0]);
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(X(from[0]), Y(from[1]));
  ctx.arc(X(from[0]), Y(from[1]), R * S, a0 - half, a0 + half);
  ctx.closePath();
  const g = ctx.createRadialGradient(X(from[0]), Y(from[1]), 0, X(from[0]), Y(from[1]), R * S);
  g.addColorStop(0, "rgba(255,240,170,0.28)"); g.addColorStop(1, "rgba(255,240,170,0.02)");
  ctx.fillStyle = g; ctx.fill();
  ctx.strokeStyle = "rgba(255,240,170,0.5)"; ctx.setLineDash([0.8 * S, 0.8 * S]); ctx.lineWidth = 0.12 * S; ctx.stroke();
  ctx.restore();
}

function draw() {
  labelQueue = [];
  drawScene();
  for (const l of labelQueue) drawLabel(...l);
}
function drawScene() {
  const mo = state.moment;
  if (!mo) { setCamera({ x: PITCH_X / 2, y: PITCH_Y / 2 }); drawPitch(); return; }
  const atPass = state.mode === "ghost" || Math.abs(state.t - passT()) < 0.051;
  const fr = state.mode === "ghost" ? frameAt(passT()) : frameAt(state.t);
  // camera target: the ball, or at the pass the middle of the passer, the pass and the option
  let target = { x: fr.ball[0], y: fr.ball[1] };
  if (atPass) {
    const pp = fr.players[mo.pass.passer], b = bestOption(), pts = [pp, mo.pass.to].concat(b ? [b.aim] : []);
    target = { x: pts.reduce((s, q) => s + q[0], 0) / pts.length, y: pts.reduce((s, q) => s + q[1], 0) / pts.length };
  }
  setCamera(target);
  drawPitch();
  const a = mo.analysis, best = bestOption(), passer = mo.pass.passer;
  const P = fr.players;
  let g = null;
  if (state.mode === "ghost") g = ghost(state.ghostT);

  if (atPass) {
    drawCone(P[passer]);
    arrow(P[passer], mo.pass.to, C("--played"), false, 0.28);
    for (const o of a.options) {
      if (o.offside && o.in_view && o.gain_vs_pass_played_pct >= 0.5)
        label(P[o.player], `${short(o.label)} offside`, C("--warn"), true);
    }
    if (best) {
      arrow(P[passer], best.aim, C("--better"), true, 0.32);
      label(best.aim, `${short(best.label)}: ${best.threat_pct}% threat, ${best.xp_pct}% to complete`, C("--better"),
            { away: P[passer] });
    }
  }

  // players
  const inView = new Set(a.options.filter((o) => o.in_view).map((o) => o.player));
  const team = (pid) => state.match.players[pid].team;
  const passerTeam = team(passer);
  P.forEach((p, pid) => {
    let pos = p;
    if (g && g.def && g.def.id === pid) pos = g.def.pos;
    if (g && g.recv && g.recv.id === pid) pos = g.recv.pos;
    const t = team(pid);
    const dim = atPass && t === passerTeam && pid !== passer && !inView.has(pid);
    ctx.save();
    ctx.globalAlpha = dim ? 0.45 : 1;
    ctx.beginPath(); ctx.arc(X(pos[0]), Y(pos[1]), 1.15 * S, 0, 2 * Math.PI);
    ctx.fillStyle = t === 0 ? C("--team0") : C("--team1"); ctx.fill();
    ctx.lineWidth = 0.18 * S; ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.stroke();
    ctx.fillStyle = "#08121c"; ctx.font = `700 ${Math.max(9, 1.15 * S)}px Inter, Segoe UI, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(state.match.players[pid].jersey, X(pos[0]), Y(pos[1]) + 0.05 * S);
    ctx.restore();
    if (pid === passer) {
      ctx.save(); ctx.strokeStyle = "#ffe680"; ctx.lineWidth = 0.3 * S;
      ctx.beginPath(); ctx.arc(X(pos[0]), Y(pos[1]), 1.9 * S, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
    }
    if (atPass && best && best.closest_defender_id === pid) {
      ctx.save(); ctx.strokeStyle = C("--warn"); ctx.lineWidth = 0.28 * S;
      ctx.beginPath(); ctx.arc(X(pos[0]), Y(pos[1]), 2.0 * S, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
    }
  });

  // ball
  const ball = g ? g.ball : fr.ball;
  ctx.save();
  ctx.beginPath(); ctx.arc(X(ball[0]), Y(ball[1]), 0.6 * S, 0, 2 * Math.PI);
  ctx.fillStyle = g ? "rgba(255,255,255,0.85)" : "#fff"; ctx.fill();
  ctx.lineWidth = 0.15 * S; ctx.strokeStyle = g ? C("--better") : "#111"; ctx.stroke();
  ctx.restore();

  if (g && g.done && best) {
    const tight = best.closest_call_s < mo.model.tight_below_s;
    label(best.aim, `ball first by ${best.closest_call_s} s${tight ? ": tight" : ""}`,
          tight ? C("--warn") : C("--better"), { away: frameAt(passT()).players[mo.pass.passer], row: 1 });
  }
}

// ---------- playback ----------
function setPlaying(on) {
  state.playing = on;
  $("playBtn").textContent = on ? "Pause" : "Play";
  state.last = null;
}
function fmt(t) {
  const mo = state.moment, abs = (mo.clip.start_frame / mo.clip.hz) + t;
  const m = Math.floor(abs / 60), s = Math.floor(abs % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
function showPassNote() {
  const best = bestOption(), a = state.moment.analysis;
  const n = $("overlayNote");
  n.hidden = false;
  n.innerHTML = best
    ? `${esc(short(a.passer))} played it to ${esc(short(a.pass_played_to))}. <b>${esc(short(best.label))}</b> was in view, onside and reachable.`
    : `${esc(short(a.passer))} played it to ${esc(short(a.pass_played_to))}. Nothing clearly better was on.`;
  $("ghostBtn").disabled = !best;
  $("continueBtn").disabled = false;
}

function tick(now) {
  const mo = state.moment;
  if (mo && state.playing) {
    const dt = state.last == null ? 0 : (now - state.last) / 1000;
    state.last = now;
    if (state.mode === "ghost") {
      state.ghostT += dt * state.speed;
      if (ghost(state.ghostT).done) setPlaying(false);
    } else {
      const end = (mo.clip.players.length - 1) / mo.clip.hz;
      const before = state.t;
      state.t = Math.min(end, state.t + dt * state.speed);
      if (!state.stoppedAtPass && before < passT() && state.t >= passT()) {
        state.t = passT(); state.stoppedAtPass = true; setPlaying(false); showPassNote();
      }
      if (state.t >= end) setPlaying(false);
    }
  }
  if (mo) {
    $("scrub").value = state.mode === "ghost" ? passT() : state.t;
    $("clock").textContent = fmt(state.mode === "ghost" ? passT() : state.t);
  }
  draw();
  requestAnimationFrame(tick);
}

$("playBtn").onclick = () => {
  if (!state.moment) return;
  if (state.mode === "ghost" && !state.playing && ghost(state.ghostT).done) { state.mode = "clip"; }
  setPlaying(!state.playing);
};
$("restartBtn").onclick = () => {
  state.mode = "clip"; state.t = 0; state.stoppedAtPass = false; $("overlayNote").hidden = true;
  $("ghostBtn").disabled = true; $("continueBtn").disabled = true; setPlaying(true);
};
$("ghostBtn").onclick = () => { state.mode = "ghost"; state.ghostT = 0; setPlaying(true); };
$("continueBtn").onclick = () => {
  state.mode = "clip"; state.t = passT() + 0.06; $("overlayNote").hidden = true; setPlaying(true);
};
$("scrub").oninput = (e) => {
  state.mode = "clip"; state.t = +e.target.value;
  state.stoppedAtPass = state.t >= passT();
  if (Math.abs(state.t - passT()) < 0.06) { state.t = passT(); showPassNote(); }
  else $("overlayNote").hidden = true;
  setPlaying(false);
};
$("speed").onchange = (e) => { state.speed = +e.target.value; };
$("zoomBtn").onclick = () => {
  state.follow = !state.follow;
  $("zoomBtn").textContent = state.follow ? "Whole pitch" : "Follow the ball";
};
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !["INPUT", "SELECT", "BUTTON", "TEXTAREA"].includes(document.activeElement.tagName)) {
    e.preventDefault(); $("playBtn").click();
  }
});
window.addEventListener("resize", resize);

// ---------- helpers ----------
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function short(label) {  // "#6 Ben Bjorn" -> "#6 Bjorn"
  if (!label) return "";
  const p = String(label).split(" ");
  return p.length >= 3 ? `${p[0]} ${p[p.length - 1]}` : label;
}

resize();
requestAnimationFrame(tick);
init().catch((e) => { $("storyBox").innerHTML = `<p class="empty">${esc(e.message)}</p>`; });
