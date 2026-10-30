"""Newton basins of the W4 method (paper Fig. 3, left).

For each initial guess on a grid, iterate the W4 map and record which of the four
solutions of  x^2 + y^2 = 4,  x^2 y = 1  it converges to (0 = none within max_iter).

Usage:  python -m static_ads.tools.basin [n_grid] [max_iter]
Output (in data/basin/): basin.dat (gnuplot matrix with x/y coordinates), basin_solutions.dat,
        basin.gp (plot script) and basin.png
"""

import shutil
import subprocess
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.optimize import fsolve

from ..paths import BASIN_DIR
from ..w4 import f_test, step

warnings.filterwarnings("ignore")  # near-singular Jacobians are expected away from solutions

DELTA_TAU = 0.5
TOL = 1e-6
LIM = 5.0


def find_solutions():
    """Four solutions of f_test, found from a coarse set of starting points."""
    sols = []
    for x0 in np.linspace(-3, 3, 7):
        for y0 in np.linspace(-3, 3, 7):
            s, info, ier, _ = fsolve(f_test, [x0, y0], full_output=True, xtol=1e-13)
            if (
                ier == 1
                and np.max(np.abs(f_test(s))) < 1e-10
                and not any(np.allclose(s, t, atol=1e-8) for t in sols)
            ):
                sols.append(s)
    sols.sort(key=lambda s: (s[0], s[1]))
    return np.array(sols)


SOLS = find_solutions()


def classify(x0, max_iter):
    """Iterate from x0; return 1-based index of the solution reached, 0 if none."""
    x = np.array(x0, dtype=float)
    p = np.zeros(2)
    for _ in range(max_iter):
        if np.max(np.abs(f_test(x))) < TOL:
            d = np.linalg.norm(SOLS - x, axis=1)
            return int(np.argmin(d)) + 1
        try:
            x, p = step(x, p, f_test, DELTA_TAU)
        except (ZeroDivisionError, np.linalg.LinAlgError):
            return 0
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e8:
            return 0
    return 0


def _work(args):
    return classify(*args)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    print("solutions:\n", SOLS)

    xs = np.linspace(-LIM, LIM, n)
    ys = np.linspace(-LIM, LIM, n)
    # Row j = y_j, column i = x_i
    jobs = [((x, y), max_iter) for y in ys for x in xs]
    with Pool() as pool:
        res = pool.map(_work, jobs, chunksize=64)
    grid = np.array(res).reshape(n, n)

    # gnuplot "nonuniform matrix": first row = x values, first column = y values
    out = np.zeros((n + 1, n + 1))
    out[0, 0] = n
    out[0, 1:] = xs
    out[1:, 0] = ys
    out[1:, 1:] = grid
    BASIN_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(BASIN_DIR / "basin.dat", out, fmt="%.10g")

    # Also save solutions for overlaying the black crosses
    np.savetxt(BASIN_DIR / "basin_solutions.dat", SOLS, header="x y", fmt="%.10g")

    with open(BASIN_DIR / "basin.gp", "w") as fh:
        fh.write(
            """set terminal pngcairo size 700,700
set output 'basin.png'
set xlabel 'x_{ini}'
set ylabel 'y_{ini}'
set xrange [-5:5]
set yrange [-5:5]
set cbrange [-0.5:4.5]
unset colorbox
# 0 = not converged (white), 1..4 = the four solutions
set palette defined (0 'white', 1 'red', 2 'orange', 3 'green', 4 'blue')
set palette maxcolors 5
set view map
set size square
set key off
splot 'basin.dat' nonuniform matrix with image, \\
      'basin_solutions.dat' using 1:2:(0) with points pt 1 ps 2 lw 2 lc 'black'
"""
        )
    counts = {k: int((grid == k).sum()) for k in range(5)}
    print("counts (0=none):", counts)
    if shutil.which("gnuplot"):
        subprocess.run(["gnuplot", "basin.gp"], cwd=BASIN_DIR, check=False)


if __name__ == "__main__":
    main()
