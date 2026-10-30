"""Build one self-contained interactive HTML page from the scan results (data/scan/scan_n*_k*.dat).

Open data/scan/scan_interactive.html in any browser (no internet, no extra packages needed):
  mouse wheel      zoom both axes about the cursor      ctrl + wheel   zoom x only
  shift + wheel    zoom y only                           drag           pan
  double click     reset that panel                      hover          values of the nearest point
  click            select that star (profile and eigenmodes in section 1)

Usage:  python -m static_ads.tools.scan_family ...       (scan data)
        python -m static_ads.tools.build_profiles        (profile/eigenmode data, once)
        python -m static_ads.tools.interactive_plot [scan_dir]
"""

import glob
import json
import os
import re
import sys

import numpy as np

from ..paths import SCAN_DIR

scan = sys.argv[1] if len(sys.argv) > 1 else str(SCAN_DIR)
data = {}
for f in sorted(glob.glob(os.path.join(scan, "scan_n*_k*.dat"))):
    m = re.search(r"scan_n([0-9.]+)_k([0-9.]+)\.dat$", f)
    if not m:
        continue
    n, k = float(m.group(1)), float(m.group(2))
    a = np.loadtxt(f)
    pts = os.path.join(scan, f"scan_n{m.group(1)}_k{m.group(2)}_points.txt")
    notes = []
    if os.path.exists(pts):
        notes = [ln.strip() for ln in open(pts) if ln.startswith(("stability_point", "extremum M", "# n="))]
    prof = os.path.join(scan, f"profiles_n{m.group(1)}_k{m.group(2)}.json")
    stars = json.load(open(prof)) if os.path.exists(prof) else []
    data[f"n={n:g}, k={k:g}"] = dict(
        stars=stars,
        n=n,
        k=k,
        rc=a[:, 0].tolist(),
        M=a[:, 1].tolist(),
        R=a[:, 2].tolist(),
        g=a[:, 3].tolist(),
        w0=a[:, 4].tolist(),
        w1=a[:, 5].tolist(),
        nneg=a[:, 6].astype(int).tolist(),
        notes=notes,
    )
order = sorted(data, key=lambda s: (data[s]["k"], data[s]["n"]))
data = {key: data[key] for key in order}

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Static stars in AdS: sequences, stability and eigenmodes (interactive)</title>
<style>
 body{font-family:system-ui,sans-serif;margin:1rem 1.5rem;color:#222;max-width:1500px}
 h1{font-size:1.35rem;margin:.2rem 0}
 h2{font-size:1.05rem;margin:1.4rem 0 .2rem;border-bottom:1px solid #ccc;padding-bottom:2px}
 .row{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px}
 .panel{position:relative}
 .panel canvas{border:1px solid #bbb;background:#fff;cursor:grab;display:block}
 .panel canvas:active{cursor:grabbing}
 .tg{font-size:11px;color:#444;margin-top:2px}
 .tg button{font-size:11px;padding:0 5px;margin-right:3px}
 .legend span{display:inline-block;margin-right:14px}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}
 #tip{position:fixed;pointer-events:none;background:rgba(30,30,30,.92);color:#fff;font-size:12px;padding:5px 8px;border-radius:4px;display:none;white-space:pre;z-index:10}
 pre{background:#f6f6f6;padding:.6rem;font-size:12px;overflow:auto;margin:.3rem 0}
 button,select{font-size:14px;padding:3px 8px}
 .hint{color:#555;font-size:13px}
 #sel{background:#fffbe6;border:1px solid #e6d98a;padding:.4rem .7rem;margin-top:6px;font-size:13px;white-space:pre}
</style></head><body>
<h1>Static polytropic stars in AdS: sequences, radial stability and eigenmodes</h1>
<div>
 Family: <select id="fam"></select>
 &nbsp; Selected star: <select id="star"></select>
 <button id="resetAll">Reset all views</button>
 <span class="hint">&nbsp; wheel: zoom &middot; ctrl+wheel: x only &middot; shift+wheel: y only &middot; drag: pan &middot; double-click: reset &middot; click a point on a sequence panel: select that star</span>
</div>
<div class="legend" style="margin-top:6px">
 <span><i class="dot" style="background:#1f4fd8"></i>stable (0 negative &omega;&sup2;)</span>
 <span><i class="dot" style="background:#d62728"></i>1 negative</span>
 <span><i class="dot" style="background:#ff9f1c"></i>2 negative</span>
 <span><i class="dot" style="background:#8e44ad"></i>3 or more negative</span>
 <span>&#9711; = selected star</span>
</div>

<h2>1. Selected star: profile and radial eigenmodes</h2>
<div>Radius coordinate for the density plots:
 <select id="rad"><option value="x">x (conformal coordinate, r = l tan x)</option><option value="r">r = l tan x (areal radius)</option></select>
 <span class="hint">&nbsp; the surface is at x_S, or r_S = tan x_S</span></div>
<div id="sel"></div>
<div class="row" id="prof"></div>
<h2>2. Sequence of equilibria (one point per star of the family)</h2>
<div class="row" id="seq"></div>
<h2>3. Turning-point diagnostic</h2>
<div class="hint">The number of negative &omega;&sup2; changes exactly where dM/d ln&rho;<sub>c</sub> changes sign (Wheeler's criterion).</div>
<div class="row" id="diag"></div>
<h2>4. Comparison of all families</h2>
<div class="row" id="cmp"></div>
<h2>Stability points and extrema of M (selected family)</h2>
<pre id="notes"></pre>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const COLORS = ['#1f4fd8', '#d62728', '#ff9f1c', '#8e44ad'];
const FAMCOL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf'];
const MODECOL = ['#d62728', '#1f4fd8', '#2ca02c', '#8e44ad'];
const col = k => COLORS[Math.min(k, 3)];
const tip = document.getElementById('tip');

function niceStep(span, target) {
  const raw = span / target, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}
function fmt(v, step) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (step) {   // enough decimals to tell neighbouring ticks apart
    const dec = Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
    if (a >= 1e6 || dec > 7) return v.toExponential(Math.min(6, Math.max(1, Math.ceil(Math.log10(a / step)))));
    return v.toFixed(dec);
  }
  return (a >= 1e5 || a < 1e-3) ? v.toExponential(1) : parseFloat(v.toPrecision(4)).toString();
}

class Panel {
  constructor(parent, cfg) {
    this.cfg = Object.assign({w: 460, h: 340, xlog: false, ylog: false}, cfg);
    const c = this.cfg;
    const div = document.createElement('div'); div.className = 'panel';
    this.cv = document.createElement('canvas');
    this.W = c.w; this.H = c.h;
    const dpr = window.devicePixelRatio || 1;
    this.cv.width = this.W * dpr; this.cv.height = this.H * dpr;
    this.cv.style.width = this.W + 'px'; this.cv.style.height = this.H + 'px';
    this.ctx = this.cv.getContext('2d'); this.ctx.scale(dpr, dpr);
    div.appendChild(this.cv);
    const tg = document.createElement('div'); tg.className = 'tg';
    for (const ax of ['x', 'y']) {
      const b = document.createElement('button'); b.textContent = 'log ' + ax;
      b.addEventListener('click', () => { c[ax + 'log'] = !c[ax + 'log']; this.reset(); this.draw(); });
      tg.appendChild(b);
    }
    div.appendChild(tg); parent.appendChild(div);
    this.m = {l: 62, r: 14, t: 28, b: 44};
    this.series = []; this.marks = [];
    this.cv.addEventListener('wheel', e => this.onWheel(e), {passive: false});
    this.cv.addEventListener('mousedown', e => this.onDown(e));
    window.addEventListener('mousemove', e => this.onMove(e));
    window.addEventListener('mouseup', e => this.onUp(e));
    this.cv.addEventListener('dblclick', () => { this.reset(); this.draw(); });
    this.cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
  }
  setSeries(series, marks) { this.series = series; this.marks = marks || []; this.reset(); this.draw(); }
  tr(v, log) { return log ? (v > 0 ? Math.log10(v) : NaN) : v; }
  reset() {
    const c = this.cfg; let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of this.series) for (let i = 0; i < s.x.length; i++) {
      const X = this.tr(s.x[i], c.xlog), Y = this.tr(s.y[i], c.ylog);
      if (!isFinite(X) || !isFinite(Y)) continue;
      x0 = Math.min(x0, X); x1 = Math.max(x1, X); y0 = Math.min(y0, Y); y1 = Math.max(y1, Y);
    }
    if (!isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
    if (c.zero && !c.ylog) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
    if (c.ylog && c.floorDec) y0 = Math.max(y0, y1 - c.floorDec);   // ignore the extreme tail near the surface
    const px = (x1 - x0 || 1) * 0.06, py = (y1 - y0 || 1) * 0.06;
    this.vx0 = x0 - px; this.vx1 = x1 + px; this.vy0 = y0 - py; this.vy1 = y1 + py;
  }
  px(x) { return this.m.l + (this.tr(x, this.cfg.xlog) - this.vx0) / (this.vx1 - this.vx0) * (this.W - this.m.l - this.m.r); }
  py(y) { return this.H - this.m.b - (this.tr(y, this.cfg.ylog) - this.vy0) / (this.vy1 - this.vy0) * (this.H - this.m.t - this.m.b); }
  ticks(lo, hi, log, len) {
    const out = [];
    if (log && hi - lo >= 1.5) {
      const st = Math.max(1, Math.ceil((hi - lo) / (len / 55)));
      for (let k = Math.ceil(lo); k <= Math.floor(hi); k += st) out.push([k, '1e' + k]);
    } else if (log) {
      const a = Math.pow(10, lo), b = Math.pow(10, hi), st = niceStep(b - a, len / 70);
      for (let v = Math.ceil(a / st) * st; v <= b; v += st) out.push([Math.log10(v), fmt(v, st)]);
    } else {
      const st = niceStep(hi - lo, len / 70);
      for (let i = Math.ceil(lo / st - 1e-9); i * st <= hi + 1e-12; i++) { const v = i * st; out.push([v, fmt(v, st)]); }
    }
    return out;
  }
  draw() {
    const g = this.ctx, c = this.cfg, m = this.m, W = this.W, H = this.H;
    g.clearRect(0, 0, W, H);
    g.font = '11px system-ui'; g.fillStyle = '#222'; g.strokeStyle = '#e3e3e3'; g.lineWidth = 1;
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    for (const [v, s] of this.ticks(this.vx0, this.vx1, c.xlog, pw)) {
      const x = m.l + (v - this.vx0) / (this.vx1 - this.vx0) * pw;
      g.beginPath(); g.moveTo(x, m.t); g.lineTo(x, H - m.b); g.stroke();
      g.textAlign = 'center'; g.fillText(s, x, H - m.b + 15);
    }
    for (const [v, s] of this.ticks(this.vy0, this.vy1, c.ylog, ph)) {
      const y = H - m.b - (v - this.vy0) / (this.vy1 - this.vy0) * ph;
      g.beginPath(); g.moveTo(m.l, y); g.lineTo(W - m.r, y); g.stroke();
      g.textAlign = 'right'; g.fillText(s, m.l - 5, y + 4);
    }
    g.strokeStyle = '#555'; g.strokeRect(m.l, m.t, pw, ph);
    g.fillStyle = '#222'; g.textAlign = 'center'; g.font = '12px system-ui';
    g.fillText(c.xlabel, m.l + pw / 2, H - 6);
    g.save(); g.translate(13, m.t + ph / 2); g.rotate(-Math.PI / 2); g.fillText(c.ylabel, 0, 0); g.restore();
    g.font = 'bold 12px system-ui'; g.textAlign = 'left'; g.fillText(c.title, m.l, 17);
    g.save(); g.beginPath(); g.rect(m.l, m.t, pw, ph); g.clip();
    if (c.zero && !c.ylog) { const y0 = this.py(0); g.strokeStyle = '#000'; g.lineWidth = 1; g.beginPath(); g.moveTo(m.l, y0); g.lineTo(W - m.r, y0); g.stroke(); }
    for (const s of this.series) {
      const n = s.x.length, lw = s.lw || (s.step ? 2 : 1.4);
      if (s.line !== false) {
        g.strokeStyle = s.color || '#999'; g.lineWidth = lw; g.beginPath(); let pen = false, py0 = 0;
        for (let i = 0; i < n; i++) {
          const x = this.px(s.x[i]), y = this.py(s.y[i]);
          if (!isFinite(x) || !isFinite(y)) { pen = false; continue; }
          if (!pen) { g.moveTo(x, y); pen = true; }
          else { if (s.step) g.lineTo(x, py0); g.lineTo(x, y); }
          py0 = y;
        }
        g.stroke();
      }
      if (s.pts !== false && !s.step) {
        const r = s.size || 3;
        for (let i = 0; i < n; i++) {
          const x = this.px(s.x[i]), y = this.py(s.y[i]); if (!isFinite(x) || !isFinite(y)) continue;
          g.fillStyle = s.colorBy ? col(s.colorBy[i]) : (s.color || '#999'); g.beginPath(); g.arc(x, y, r, 0, 6.2832); g.fill();
        }
      }
    }
    for (const mk of this.marks) {
      const x = this.px(mk.x), y = this.py(mk.y); if (!isFinite(x) || !isFinite(y)) continue;
      g.strokeStyle = '#000'; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 7, 0, 6.2832); g.stroke();
    }
    g.restore();
    if (c.legend) {   // legend box (top right)
      let y = m.t + 12; g.font = '11px system-ui'; g.textAlign = 'left';
      const ent = this.series.filter(s => s.label);
      const wmax = Math.max(...ent.map(s => g.measureText(s.label).width), 0) + 28;
      g.fillStyle = 'rgba(255,255,255,.85)'; g.fillRect(W - m.r - wmax - 4, m.t + 2, wmax + 2, ent.length * 15 + 4);
      for (const s of ent) {
        g.fillStyle = s.color || '#999'; g.fillRect(W - m.r - wmax, y - 7, 14, 4);
        g.fillStyle = '#222'; g.fillText(s.label, W - m.r - wmax + 18, y); y += 15;
      }
    }
  }
  pos(e) { const r = this.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  onWheel(e) {
    e.preventDefault();
    const [mx, my] = this.pos(e), pw = this.W - this.m.l - this.m.r, ph = this.H - this.m.t - this.m.b;
    const fx = (mx - this.m.l) / pw, fy = 1 - (my - this.m.t) / ph;
    const k = Math.exp(e.deltaY * 0.0015);
    const cx = this.vx0 + fx * (this.vx1 - this.vx0), cy = this.vy0 + fy * (this.vy1 - this.vy0);
    if (!e.shiftKey) { this.vx0 = cx - (cx - this.vx0) * k; this.vx1 = cx + (this.vx1 - cx) * k; }
    if (!e.ctrlKey) { this.vy0 = cy - (cy - this.vy0) * k; this.vy1 = cy + (this.vy1 - cy) * k; }
    this.draw();
  }
  onDown(e) { this.drag = {p: this.pos(e), v: [this.vx0, this.vx1, this.vy0, this.vy1], moved: false}; }
  onUp(e) {
    const d = this.drag; this.drag = null;
    if (d && !d.moved && this.cfg.onPick) {
      const hit = this.nearest(d.p[0], d.p[1], 14);
      if (hit) this.cfg.onPick(hit.s, hit.i);
    }
  }
  nearest(mx, my, rad) {
    let best = null, bd = rad * rad;
    for (const s of this.series) {
      if (!s.info && !s.pick) continue;
      for (let i = 0; i < s.x.length; i++) {
        const dx = this.px(s.x[i]) - mx, dy = this.py(s.y[i]) - my, d2 = dx * dx + dy * dy;
        if (d2 < bd) { bd = d2; best = {s, i}; }
      }
    }
    return best;
  }
  onMove(e) {
    if (this.drag) {
      const [mx, my] = this.pos(e), pw = this.W - this.m.l - this.m.r, ph = this.H - this.m.t - this.m.b;
      if (Math.abs(mx - this.drag.p[0]) + Math.abs(my - this.drag.p[1]) > 3) this.drag.moved = true;
      if (!this.drag.moved) return;
      const dx = (mx - this.drag.p[0]) / pw * (this.drag.v[1] - this.drag.v[0]);
      const dy = (my - this.drag.p[1]) / ph * (this.drag.v[3] - this.drag.v[2]);
      this.vx0 = this.drag.v[0] - dx; this.vx1 = this.drag.v[1] - dx;
      this.vy0 = this.drag.v[2] + dy; this.vy1 = this.drag.v[3] + dy;
      this.draw(); tip.style.display = 'none'; return;
    }
    const r = this.cv.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
    const [mx, my] = this.pos(e), hit = this.nearest(mx, my, 12);
    if (!hit || !hit.s.info) { tip.style.display = 'none'; return; }
    tip.textContent = hit.s.info(hit.i);
    tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; tip.style.display = 'block';
  }
}

// ---------------------------------------------------------------- state and panels
let fam = null, famKey = null, selIdx = 0;
const all = [];
const mk = (host, cfg) => { const p = new Panel(document.getElementById(host), cfg); all.push(p); return p; };
const pickSeq = (s, i) => { if (fam) selectByRc(fam.rc[i]); };
const P = {
  mr:   mk('seq', {title: 'Mass-radius curve', xlabel: 'x_S (radius)', ylabel: 'M', onPick: pickSeq}),
  mrc:  mk('seq', {title: 'Mass vs central density', xlabel: 'rho_c', ylabel: 'M', xlog: true, onPick: pickSeq}),
  rrc:  mk('seq', {title: 'Radius vs central density', xlabel: 'rho_c', ylabel: 'x_S', xlog: true, onPick: pickSeq}),
  gS:   mk('seq', {title: 'Surface value g(x_S)', xlabel: 'rho_c', ylabel: 'g(x_S)', xlog: true, onPick: pickSeq}),
  w2:   mk('seq', {title: 'Two lowest omega^2', xlabel: 'rho_c', ylabel: 'omega^2', xlog: true, zero: true, legend: true, onPick: pickSeq}),
  nneg: mk('seq', {title: 'Number of unstable modes', xlabel: 'rho_c', ylabel: 'negative omega^2', xlog: true, onPick: pickSeq}),
  dM:   mk('diag', {title: 'dM / d ln rho_c', xlabel: 'rho_c', ylabel: 'dM/dln(rho_c)', xlog: true, zero: true, onPick: pickSeq}),
  w0z:  mk('diag', {title: 'Lowest omega^2 vs dM/dln(rho_c)', xlabel: 'dM/dln(rho_c)', ylabel: 'lowest omega^2', zero: true, onPick: pickSeq}),
  rho:  mk('prof', {title: 'Density profile of the selected star', xlabel: 'x', ylabel: 'rho'}),
  rhoAll: mk('prof', {title: 'Density profiles of all stored stars of the family', xlabel: 'x', ylabel: 'rho', ylog: true, floorDec: 18, legend: true, w: 560,
                      onPick: s => { if (s.starIdx !== undefined) { selIdx = s.starIdx; starSel.value = selIdx; renderFamily(); renderStar(); } }}),
  gp:   mk('prof', {title: 'Metric function g(x)', xlabel: 'x', ylabel: 'g(x)'}),
  xi:   mk('prof', {title: 'Radial eigenmodes xi_i(x) (normalised)', xlabel: 'x', ylabel: 'xi', zero: true, legend: true, w: 560}),
  cMR:  mk('cmp', {title: 'Mass-radius curves, all families', xlabel: 'x_S', ylabel: 'M', legend: true}),
  cN:   mk('cmp', {title: 'Unstable modes, all families', xlabel: 'rho_c', ylabel: 'negative omega^2', xlog: true, legend: true}),
  cW:   mk('cmp', {title: 'Lowest omega^2, all families', xlabel: 'rho_c', ylabel: 'omega^2', xlog: true, zero: true, legend: true}),
  cM:   mk('cmp', {title: 'Mass vs central density, all families', xlabel: 'rho_c', ylabel: 'M', xlog: true, legend: true}),
};
P.rho.cfg.ylog = false;
let useAreal = false;
const radius = arr => useAreal ? arr.map(Math.tan) : arr;
function setRadiusLabels() { const l = useAreal ? 'r = tan x' : 'x'; P.rho.cfg.xlabel = l; P.rhoAll.cfg.xlabel = l; P.gp.cfg.xlabel = l; }

function dMdlnrc(d) {
  const n = d.rc.length, out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(i - 1, 0), b = Math.min(i + 1, n - 1);
    out[i] = (d.M[b] - d.M[a]) / (Math.log(d.rc[b]) - Math.log(d.rc[a]));
  }
  return out;
}
const info = d => i => `rho_c = ${d.rc[i].toPrecision(6)}\nM = ${d.M[i].toPrecision(8)}\nx_S = ${d.R[i].toPrecision(7)}\n` +
  `g(x_S) = ${d.g[i].toPrecision(7)}\nomega^2 (lowest) = ${d.w0[i].toPrecision(5)}\nomega^2 (2nd) = ${d.w1[i].toPrecision(5)}\n` +
  `negative eigenvalues: ${d.nneg[i]}\n(click to select)`;

function selMarks(d) {
  const st = d.stars[selIdx]; if (!st) return [];
  let j = 0, best = 1e9;
  for (let i = 0; i < d.rc.length; i++) { const e = Math.abs(Math.log(d.rc[i] / st.rc)); if (e < best) { best = e; j = i; } }
  return j;
}
function renderFamily() {
  const d = fam; d.dM = d.dM || dMdlnrc(d);
  const j = selMarks(d), has = typeof j === 'number';
  const mark = (xs, ys) => has ? [{x: xs[j], y: ys[j]}] : [];
  const base = (x, y, extra) => Object.assign({x, y, colorBy: d.nneg, color: '#aaa', info: info(d), size: 3}, extra || {});
  P.mr.setSeries([base(d.R, d.M)], mark(d.R, d.M));
  P.mrc.setSeries([base(d.rc, d.M)], mark(d.rc, d.M));
  P.rrc.setSeries([base(d.rc, d.R)], mark(d.rc, d.R));
  P.gS.setSeries([base(d.rc, d.g)], mark(d.rc, d.g));
  P.w2.setSeries([base(d.rc, d.w0, {label: 'lowest'}), {x: d.rc, y: d.w1, color: '#888', label: 'second', info: info(d), pts: true, size: 2}], mark(d.rc, d.w0));
  P.nneg.setSeries([{x: d.rc, y: d.nneg, step: true, color: '#222', info: info(d)}], mark(d.rc, d.nneg));
  P.dM.setSeries([base(d.rc, d.dM)], mark(d.rc, d.dM));
  P.w0z.setSeries([base(d.dM, d.w0)], mark(d.dM, d.w0));
  document.getElementById('notes').textContent = d.notes.join('\n');
}
function renderStar() {
  const st = fam.stars[selIdx];
  if (!st) { document.getElementById('sel').textContent = 'no stored profile for this family'; return; }
  document.getElementById('sel').textContent =
    `rho_c = ${st.rc.toPrecision(6)}    M = ${st.M.toPrecision(7)}    x_S = ${st.R.toPrecision(6)}    ` +
    `negative eigenvalues: ${st.nneg} (${st.nneg ? 'UNSTABLE' : 'stable'})\nomega^2 of the 4 lowest modes: ` + st.w2.map(v => v.toPrecision(5)).join(',  ');
  setRadiusLabels();
  P.rho.setSeries([{x: radius(st.x), y: st.rho, color: '#1f77b4', pts: false}]);
  const stars = fam.stars.map((s, i) => ({x: radius(s.x), y: s.rho, color: col(s.nneg), pts: false, lw: 1.3, pick: true, starIdx: i,
    info: () => `rho_c = ${s.rc.toPrecision(5)}\nM = ${s.M.toPrecision(6)}\nx_S = ${s.R.toPrecision(5)}\n${s.nneg ? s.nneg + ' unstable mode(s)' : 'stable'}\n(click to select)`}));
  stars.push({x: radius(st.x), y: st.rho, color: '#000', pts: false, lw: 3.2, label: `selected: rho_c=${st.rc.toPrecision(3)}`});
  P.rhoAll.setSeries(stars);
  P.gp.setSeries([{x: radius(st.x), y: st.g, color: '#d62728', pts: false}]);
  P.xi.setSeries(st.xi.map((v, i) => ({x: st.x, y: v, color: MODECOL[i], pts: false, label: `mode ${i}: w^2=${st.w2[i].toPrecision(3)}`})));
}
function selectByRc(rc) {
  let bi = 0, be = 1e9;
  fam.stars.forEach((s, i) => { const e = Math.abs(Math.log(s.rc / rc)); if (e < be) { be = e; bi = i; } });
  selIdx = bi; document.getElementById('star').value = bi; renderFamily(); renderStar();
}
function renderCompare() {
  const keys = Object.keys(DATA);
  P.cMR.setSeries(keys.map((k, i) => ({x: DATA[k].R, y: DATA[k].M, color: FAMCOL[i % 8], label: k, size: 2})));
  P.cN.setSeries(keys.map((k, i) => ({x: DATA[k].rc, y: DATA[k].nneg, color: FAMCOL[i % 8], label: k, step: true})));
  P.cW.setSeries(keys.map((k, i) => ({x: DATA[k].rc, y: DATA[k].w0, color: FAMCOL[i % 8], label: k, size: 2})));
  P.cM.setSeries(keys.map((k, i) => ({x: DATA[k].rc, y: DATA[k].M, color: FAMCOL[i % 8], label: k, size: 2})));
}
document.getElementById('rad').addEventListener('change', e => { useAreal = e.target.value === 'r'; if (fam) renderStar(); });
const famSel = document.getElementById('fam'), starSel = document.getElementById('star');
for (const key of Object.keys(DATA)) { const o = document.createElement('option'); o.value = o.textContent = key; famSel.appendChild(o); }
function chooseFamily() {
  famKey = famSel.value; fam = DATA[famKey]; fam.stars = fam.stars || [];
  starSel.innerHTML = '';
  fam.stars.forEach((s, i) => { const o = document.createElement('option'); o.value = i; o.textContent = `rho_c = ${s.rc.toPrecision(4)}  (${s.nneg ? s.nneg + ' unstable' : 'stable'})`; starSel.appendChild(o); });
  // start at the first stored star past the first stability point, otherwise the first one
  selIdx = Math.min(1, Math.max(fam.stars.length - 1, 0));
  starSel.value = selIdx; renderFamily(); renderStar();
}
famSel.addEventListener('change', chooseFamily);
starSel.addEventListener('change', () => { selIdx = +starSel.value; renderFamily(); renderStar(); });
document.getElementById('resetAll').addEventListener('click', () => all.forEach(p => { p.reset(); p.draw(); }));
renderCompare();
if (Object.keys(DATA).length) { famSel.value = Object.keys(DATA).includes('n=5, k=1') ? 'n=5, k=1' : Object.keys(DATA)[0]; chooseFamily(); }
</script></body></html>
"""
out = os.path.join(scan, "scan_interactive.html")
open(out, "w").write(HTML.replace("__DATA__", json.dumps(data)))
print(f"wrote {out} with {len(data)} families: {', '.join(data)}")
