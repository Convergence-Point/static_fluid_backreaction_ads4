"""Radial-stability analysis of every stored solution under data/runs/*/.

For each run folder the converged profile (x, g, rho) is read from star_profile.dat, the unknown
vector (g, w = rho^(1/n), x_S) is rebuilt and the eigenproblem  M eta = w^2 eta  is analysed:
the exact number of negative eigenvalues (Sturm count) and the two lowest eigenvalues.
A solution is unstable if it has at least one negative w^2.

Usage:  python -m static_ads.tools.analyze_runs [runs_dir] [out_dir]

Outputs (in out_dir, default data/analysis/):
  stability_all.dat       one line per run: n k N rho_c M x_S w2_0 w2_1 n_negative stable
  stability_all.gp/.png   mass-radius curves coloured by stability, one panel per n
  stability_turning.txt   check of the turning-point criterion per (n, k) family
"""

import glob
import os
import shutil
import subprocess
import sys
import warnings

import numpy as np

from ..paths import ANALYSIS_DIR, RUNS_DIR
from ..stability import lowest_eigenvalues, stability
from ..star import Star, mass_from_g, pack

warnings.filterwarnings("ignore")
root = sys.argv[1] if len(sys.argv) > 1 else str(RUNS_DIR)
out_dir = sys.argv[2] if len(sys.argv) > 2 else str(ANALYSIS_DIR)
os.makedirs(out_dir, exist_ok=True)

rows = []
for d in sorted(glob.glob(os.path.join(root, "*", "star_profile.dat"))):
    folder = os.path.dirname(d)
    try:
        n, k, N_s, rho_c, M, x_S, g_S, iters, conv = np.loadtxt(os.path.join(folder, "summary.dat")).tolist()
    except Exception:
        print("skipping (no summary):", folder)
        continue
    data = np.loadtxt(d)
    x, g, rho = data[:, 0], data[:, 1], data[:, 2]
    N = len(x) - 1
    star = Star(n=n, k=k, N=N, rho_c=rho_c)
    w = np.abs(rho) ** (1.0 / n)
    w[-1] = 0.0
    u = pack(g, w, x[-1])
    stable, w2_min, n_neg = stability(star, u, x=x)
    w2 = lowest_eigenvalues(star, u, 2, x=x)
    rows.append((n, k, N, rho_c, mass_from_g(g[-1], x[-1]), x[-1], w2[0], w2[1], n_neg, int(stable)))
    print(
        f"n={n:g} k={k:g} N={N:5d} rho_c={rho_c:10.5g} M={rows[-1][4]:.5g} -> "
        f"{'stable  ' if stable else 'UNSTABLE'} (negative eigenvalues: {n_neg}, lowest w^2 = {w2[0]:.4g})"
    )

rows = np.array(rows)
out = os.path.join(out_dir, "stability_all.dat")
np.savetxt(out, rows, header="n k N rho_c M x_S w2_0 w2_1 n_negative stable", fmt="%.10g")
print(
    f"\n{len(rows)} solutions analysed -> {out}; "
    f"{int(rows[:, 9].sum())} stable, {int((1 - rows[:, 9]).sum())} unstable"
)

# ---- families: pool runs per (n, k), keep the highest N for each rho_c
fam = {}
for r in rows:
    key = (r[0], r[1])
    fam.setdefault(key, {})
    rc_key = round(float(r[3]), 6)
    if rc_key not in fam[key] or r[2] > fam[key][rc_key][2]:
        fam[key][rc_key] = r
turning = []
for key, d in sorted(fam.items()):
    arr = np.array([d[q] for q in sorted(d)])
    if len(arr) < 4:
        continue
    rc, M, nneg = arr[:, 3], arr[:, 4], arr[:, 8]
    ext = [i for i in range(1, len(M) - 1) if (M[i] - M[i - 1]) * (M[i + 1] - M[i]) < 0]
    jumps = [i for i in range(1, len(nneg)) if nneg[i] != nneg[i - 1]]
    turning.append(
        f"n={key[0]:g} k={key[1]:g}: {len(arr)} solutions, rho_c in [{rc[0]:.3g}, {rc[-1]:.3g}]\n"
        f"   extrema of M(rho_c) at rho_c ~ {[f'{(rc[i]):.3g}' for i in ext]}\n"
        f"   n_negative changes between rho_c = {[f'{rc[i-1]:.3g}..{rc[i]:.3g} ({int(nneg[i-1])}->{int(nneg[i])})' for i in jumps]}"
    )
with open(os.path.join(out_dir, "stability_turning.txt"), "w") as fh:
    fh.write("\n".join(turning) + "\n")
print("\n".join(turning))

# ---- plot: M-R curves coloured by stability, one panel per n (k = 1 families first)
ns = sorted({key for key in fam})
panels = []
for n, k in ns:
    d = fam[(n, k)]
    arr = np.array([d[q] for q in sorted(d)])
    fn = f"stability_curve_n{n:g}_k{k:g}.dat"
    np.savetxt(
        os.path.join(out_dir, fn), arr[:, [3, 4, 5, 6, 8]], header="rho_c M x_S w2_0 n_negative", fmt="%.10g"
    )
    panels.append((n, k, fn, len(arr)))
panels = [p for p in panels if p[3] >= 4]
cols = 3
rows_n = (len(panels) + cols - 1) // cols
with open(os.path.join(out_dir, "stability_all.gp"), "w") as fh:
    fh.write(f"set terminal pngcairo size {450*cols},{380*rows_n}\nset output 'stability_all.png'\n")
    fh.write(
        f"set multiplot layout {rows_n},{cols} title 'Radial stability of all solutions (blue: stable, red: unstable)'\n"
    )
    fh.write("set grid\nset xlabel 'x_S'\nset ylabel 'M'\nset key off\n")
    for n, k, fn, _ in panels:
        fh.write(f"set title 'n={n:g}, k={k:g}'\n")
        fh.write(
            f"plot '{fn}' using 3:2 with lines lc rgb 'gray', "
            f"'' using ($5==0 ? $3 : 1/0):2 with points pt 7 lc rgb 'blue', "
            f"'' using ($5>0 ? $3 : 1/0):2 with points pt 7 lc rgb 'red'\n"
        )
    fh.write("unset multiplot\n")
if shutil.which("gnuplot"):
    subprocess.run(["gnuplot", "stability_all.gp"], cwd=out_dir, check=False)
