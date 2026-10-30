"""Run star.py over a grid of parameters; one folder per run under data/runs/.

Examples:
  python -m static_ads.tools.sweep --n 1 --k 1 --rho-range 0.01 3 30 --log --tag rho_scan
  python -m static_ads.tools.sweep --n 0.5 1 1.5 --k 1 --mass 1e-3
  python -m static_ads.tools.sweep --n 1 --k 0.5 1 2 --rho-c 1 --tag k_scan

Plots and curve data go to data/sweeps/: compare_<tag>.png overlays rho(x) and g(x) of the runs of
the sweep, curve_<tag>.png shows M, x_S and the stability along the sequence.
"""

import argparse
import itertools
import os
import shutil
import subprocess
import warnings

from ..paths import RUNS_DIR, SWEEPS_DIR
from ..star import continue_rho, run, run_dir

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n", type=float, nargs="+", default=[1.0])
ap.add_argument("--k", type=float, nargs="+", default=[1.0])
bc = ap.add_mutually_exclusive_group()
bc.add_argument("--rho-c", type=float, nargs="+")
bc.add_argument("--mass", type=float, nargs="+")
bc.add_argument(
    "--rho-range",
    type=float,
    nargs=3,
    metavar=("LO", "HI", "NUM"),
    help="NUM central densities between LO and HI (add --log for geometric spacing)",
)
ap.add_argument("--log", action="store_true", help="geometric spacing for --rho-range")
ap.add_argument("--plot-each", action="store_true", help="also make star.png in every run folder")
ap.add_argument("--N", type=int, default=1000)
ap.add_argument("--root", default=str(RUNS_DIR), help="where the per-star folders go")
ap.add_argument("--out", default=str(SWEEPS_DIR), help="where the sweep plots go")
ap.add_argument("--tag", default="sweep")
a = ap.parse_args()
import numpy as np

os.makedirs(a.out, exist_ok=True)
rel_root = os.path.relpath(a.root, a.out)  # run folders as seen from the plot scripts

if a.rho_range:
    lo, hi, num = a.rho_range
    values = list(np.geomspace(lo, hi, int(num)) if a.log else np.linspace(lo, hi, int(num)))
elif a.mass:
    values = a.mass
else:
    values = a.rho_c if a.rho_c else [1.0]
use_mass = bool(a.mass)
plot_each = a.plot_each or len(values) * len(a.n) * len(a.k) <= 12

dirs = []
curves = {}  # (n, k) -> rows (rho_c, M, x_S, g_S)
for n, k in itertools.product(a.n, a.k):
    prev_u, prev_v = None, None  # continuation in rho_c: previous solution is the next guess
    for v in sorted(values) if not use_mass else values:
        res = None
        if prev_u is not None and not use_mass:
            u_to, ok = continue_rho(n, k, a.N, prev_u, prev_v, v)
            if ok:
                res = run(n=n, k=k, N=a.N, rho_c=v, root=a.root, plot=plot_each, guess=u_to)
        if res is None or not res["converged"]:
            res = run(
                n=n,
                k=k,
                N=a.N,
                rho_c=None if use_mass else v,
                mass=v if use_mass else None,
                root=a.root,
                plot=plot_each,
            )
        if res["converged"]:
            prev_u, prev_v = res["u"], v
            dirs.append((n, k, v, os.path.basename(run_dir(res["star"], a.root))))
            curves.setdefault((n, k), []).append(
                (
                    res["rho_c"],
                    res["M"],
                    res["x_S"],
                    res["g_S"],
                    np.nan if res["w2_min"] is None else res["w2_min"],
                    -1 if res["n_neg"] is None else res["n_neg"],
                )
            )

# One data file per (n, k) with the whole family, plus a plot of M and x_S versus rho_c.
if curves and (a.rho_range or len(values) > 3):
    plots_m, plots_r, plots_mr = [], [], []
    for (n, k), rows in curves.items():
        fn = f"curve_{a.tag}_n{n:g}_k{k:g}.dat"
        np.savetxt(
            os.path.join(a.out, fn), np.array(rows), header="rho_c M x_S g_S w2_min n_negative", fmt="%.10g"
        )
        lab = f"n={n:g} k={k:g}"
        plots_m.append(f"'{fn}' using 1:2 with linespoints title '{lab}'")
        plots_r.append(f"'{fn}' using 1:3 with linespoints title '{lab}'")
        plots_mr.append(
            f"'{fn}' using ($6==0 ? $3 : 1/0):2 with linespoints pt 7 lc rgb 'blue' title '{lab} stable', "
            f"'' using ($6>0 ? $3 : 1/0):2 with linespoints pt 7 lc rgb 'red' title '{lab} unstable'"
        )
    with open(os.path.join(a.out, f"curve_{a.tag}.gp"), "w") as fh:
        fh.write(
            f"""set terminal pngcairo size 1400,450
set output 'curve_{a.tag}.png'
set multiplot layout 1,3 title 'Stellar family: {a.tag}'
set grid
set logscale x
set xlabel 'rho_c'
set ylabel 'M'
set key bottom right
plot {", ".join(plots_m)}
set ylabel 'x_S'
set key top right
plot {", ".join(plots_r)}
unset logscale x
set xlabel 'x_S (radius)'
set ylabel 'M'
set key bottom right
plot {", ".join(plots_mr)}
unset multiplot
"""
        )
    if shutil.which("gnuplot"):
        subprocess.run(["gnuplot", f"curve_{a.tag}.gp"], cwd=a.out, check=False)
    print(f"wrote {a.out}/curve_{a.tag}.gp")

if dirs and len(dirs) <= 12:

    def label(n, k, v):
        return f"n={n:g} k={k:g} " + (f"M={v:g}" if use_mass else f"rho_c={v:g}")

    rho_plots = ", \\\n     ".join(
        f"'{rel_root}/{d}/star_profile.dat' using 1:3 with lines lw 2 title '{label(n, k, v)}'"
        for n, k, v, d in dirs
    )
    g_plots = ", \\\n     ".join(
        f"'{rel_root}/{d}/star_profile.dat' using 1:2 with lines lw 2 title '{label(n, k, v)}'"
        for n, k, v, d in dirs
    )
    path = os.path.join(a.out, f"compare_{a.tag}.gp")
    with open(path, "w") as fh:
        fh.write(
            f"""set terminal pngcairo size 1100,450
set output 'compare_{a.tag}.png'
set multiplot layout 1,2 title 'Comparison: {a.tag}'
set xlabel 'x'
set grid
set logscale y
set ylabel 'rho(x)'
set key top right
plot {rho_plots}
unset logscale y
set ylabel 'g(x)'
set key top left
plot {g_plots}
unset multiplot
"""
        )
    if shutil.which("gnuplot"):
        subprocess.run(["gnuplot", f"compare_{a.tag}.gp"], cwd=a.out, check=False)
    print(f"wrote {path}")
