"""Follow a family of stars to very large central density and locate the stability changes.

For every (n, k) the central density rho_c is raised geometrically from --rho-min to --rho-max
(continuation: each W4 solve starts from the previous solution).  For every star the stability
(exact number of negative eigenvalues of M, lowest two eigenvalues) is computed.  Then

  * every interval in which the number of negative eigenvalues changes is bisected in ln(rho_c)
    until it is narrower than --tol (default 1e-4 relative), and a band of extra points is added
    on both sides of every such stability point;
  * every interval containing an extremum of M(rho_c) is filled with extra points.

A graded mesh (x = x_S s^grade, default 2) resolves the small core of high-density stars.

Outputs (in --out, default data/scan/):
  scan_n<n>_k<k>.dat          rho_c M x_S g_S w2_0 w2_1 n_negative   (sorted by rho_c)
  scan_n<n>_k<k>_points.txt   stability points (n_negative changes) and extrema of M
  scan_n<n>_k<k>.gp/.png/.svg M-R curve coloured by stability, lowest w^2, n_negative
  scan_overview.gp/.png/.svg  n_negative(rho_c) for all families

Usage:  python -m static_ads.tools.scan_family --n 1 1.5 3 5 --k 1 --rho-max 1e7 --N 2000
"""

import argparse
import os
import shutil
import subprocess
import warnings
from multiprocessing import Pool

import numpy as np

from ..paths import SCAN_DIR
from ..stability import lowest_eigenvalues, stability
from ..star import Star, continue_rho, mass_from_g, solve_dummy, unpack

warnings.filterwarnings("ignore")


class Family:
    def __init__(self, n, k, N, grade, verbose=True):
        self.n, self.k, self.N, self.grade = n, k, N, grade
        self.pts = {}  # rho_c -> (u, record)
        self.verbose = verbose

    def label(self):
        return f"n={self.n:g} k={self.k:g}"

    def star(self, rc):
        return Star(n=self.n, k=self.k, N=self.N, rho_c=rc, grade=self.grade)

    def record(self, rc, u):
        st = self.star(rc)
        g, w, xs = unpack(u)
        stable, w2min, nneg = stability(st, u)
        w2 = lowest_eigenvalues(st, u, 2)
        rec = (rc, mass_from_g(g[-1], xs), xs, g[-1], w2[0], w2[1], nneg)
        self.pts[rc] = (u, rec)
        return rec

    def nearest(self, rc):
        keys = np.array(sorted(self.pts))
        return keys[np.argmin(np.abs(np.log(keys / rc)))]

    def add(self, rc):
        """Solve at rc by continuation from the nearest known point; returns the record or None."""
        if rc in self.pts:
            return self.pts[rc][1]
        r0 = self.nearest(rc)
        u, ok = continue_rho(self.n, self.k, self.N, self.pts[r0][0], r0, rc, grade=self.grade)
        if not ok:
            return None
        return self.record(rc, u)

    def start(self, rho_min):
        for rc0 in (rho_min, 0.3, 1.0, 0.1, 3.0):
            st = self.star(rc0)
            u, _, ok = solve_dummy(st, rc0)
            if ok:
                self.record(rc0, u)
                if rc0 != rho_min:
                    if self.add(rho_min) is None:
                        continue
                return True
        return False

    def march(self, rho_max, ratio):
        rc = min(self.pts)
        top = max(self.pts)
        while top < rho_max:
            nxt = min(top * ratio, rho_max)
            if self.add(nxt) is None:
                if self.verbose:
                    print(f"[{self.label()}] continuation stopped at rho_c ~ {top:.3g}", flush=True)
                return False
            top = nxt
            if self.verbose and len(self.pts) % 10 == 0:
                print(f"[{self.label()}] reached rho_c = {top:.3g} ({len(self.pts)} points)", flush=True)
        return True

    def sorted_records(self):
        return [self.pts[r][1] for r in sorted(self.pts)]

    def refine_transitions(self, tol, band, band_ratio):
        """Bisect each interval where n_negative changes, then add a band of points around it."""
        done = []
        changed = True
        while changed:
            changed = False
            recs = self.sorted_records()
            for a, b in zip(recs[:-1], recs[1:]):
                if a[6] != b[6] and b[0] / a[0] - 1 > tol:
                    mid = np.sqrt(a[0] * b[0])
                    if self.add(mid) is not None:
                        changed = True
                    else:  # cannot reach the midpoint: stop refining here
                        done.append((a[0], b[0]))
        recs = self.sorted_records()
        for a, b in zip(recs[:-1], recs[1:]):
            if a[6] != b[6]:
                centre = np.sqrt(a[0] * b[0])
                for j in range(-band, band + 1):
                    self.add(centre * band_ratio**j)

    def refine_extrema(self, fill):
        recs = self.sorted_records()
        M = np.array([r[1] for r in recs])
        rc = np.array([r[0] for r in recs])
        for i in range(1, len(recs) - 1):
            if (M[i] - M[i - 1]) * (M[i + 1] - M[i]) < 0:
                lo, hi = rc[i - 1], rc[i + 1]
                for q in np.geomspace(lo, hi, fill + 2)[1:-1]:
                    self.add(q)

    def points_report(self):
        recs = self.sorted_records()
        rc = np.array([r[0] for r in recs])
        M = np.array([r[1] for r in recs])
        R = np.array([r[2] for r in recs])
        nneg = np.array([r[6] for r in recs])
        w0 = np.array([r[4] for r in recs])
        lines = [
            f"# {self.label()}  N={self.N} grade={self.grade:g}  ({len(recs)} points, rho_c {rc[0]:.3g} .. {rc[-1]:.3g})"
        ]
        lines.append(
            "# stability points (n_negative changes); rho_c* from linear interpolation of the lowest/second eigenvalue"
        )
        for i in range(1, len(recs)):
            if nneg[i] != nneg[i - 1]:
                a, b = recs[i - 1], recs[i]
                idx = int(min(nneg[i - 1], nneg[i]))  # 0-based index of the eigenvalue that crosses zero
                f = []
                for r in (a, b):
                    st = self.star(r[0])
                    f.append(lowest_eigenvalues(st, self.pts[r[0]][0], idx + 1)[idx])
                t = f[0] / (f[0] - f[1]) if f[0] != f[1] else 0.5
                t = min(max(t, 0.0), 1.0)
                rstar = a[0] * (b[0] / a[0]) ** t
                lines.append(
                    f"stability_point rho_c*={rstar:.6g} bracket=[{a[0]:.6g},{b[0]:.6g}] "
                    f"n_negative {int(nneg[i-1])}->{int(nneg[i])} (eigenvalue #{idx + 1} crosses zero) "
                    f"M={a[1] + t*(b[1]-a[1]):.7f} x_S={a[2] + t*(b[2]-a[2]):.6f}"
                )
        lines.append("# extrema of M(rho_c) and of the radius x_S(rho_c)")
        for name, y in (("M", M), ("x_S", R)):
            for i in range(1, len(y) - 1):
                if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0:
                    kind = "max" if y[i] > y[i - 1] else "min"
                    lines.append(f"extremum {name} {kind} near rho_c={rc[i]:.6g} (grid) value={y[i]:.8f}")
        return lines


def make_svg(out, base):
    """Vector (SVG) copy of a gnuplot script's figure, so it can be zoomed without losing detail."""
    import re

    txt = open(os.path.join(out, base + ".gp")).read()
    txt = re.sub(
        r"set terminal pngcairo size (\d+),(\d+)",
        r"set terminal svg size \1,\2 dynamic enhanced font 'Helvetica,11'",
        txt,
        count=1,
    )
    txt = re.sub(r"set output '.*\.png'", f"set output '{base}.svg'", txt, count=1)
    tmp = os.path.join(out, base + "_svg.gp")
    open(tmp, "w").write(txt)
    subprocess.run(["gnuplot", base + "_svg.gp"], cwd=out, check=False)
    os.remove(tmp)


def run_family(args):
    n, k, a = args
    fam = Family(n, k, a.N, a.grade)
    if not fam.start(a.rho_min):
        return n, k, None, [f"# n={n:g} k={k:g}: could not obtain a starting solution"]
    fam.march(a.rho_max, a.ratio)
    fam.refine_transitions(a.tol, a.band, a.band_ratio)
    fam.refine_extrema(a.fill)
    fam.refine_transitions(a.tol, 0, a.band_ratio)  # transitions possibly created by the new points
    recs = np.array(fam.sorted_records())
    os.makedirs(a.out, exist_ok=True)
    tag = f"n{n:g}_k{k:g}"
    np.savetxt(
        os.path.join(a.out, f"scan_{tag}.dat"),
        recs,
        header="rho_c M x_S g_S w2_0 w2_1 n_negative",
        fmt="%.10g",
    )
    rep = fam.points_report()
    with open(os.path.join(a.out, f"scan_{tag}_points.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")
    with open(os.path.join(a.out, f"scan_{tag}.gp"), "w") as fh:
        fh.write(
            f"""set terminal pngcairo size 1500,450
set output 'scan_{tag}.png'
set multiplot layout 1,3 title 'n={n:g}, k={k:g}: radial stability along the sequence (N={a.N}, grade {a.grade:g})'
set grid
set xlabel 'x_S'
set ylabel 'M'
set key top right
plot 'scan_{tag}.dat' using 3:2 with lines lc rgb 'gray' notitle, \\
     '' using ($7==0 ? $3 : 1/0):2 with points pt 7 ps 0.6 lc rgb 'blue' title 'stable (0 negative)', \\
     '' using ($7==1 ? $3 : 1/0):2 with points pt 7 ps 0.6 lc rgb 'red' title 'unstable: 1 negative', \\
     '' using ($7==2 ? $3 : 1/0):2 with points pt 7 ps 0.6 lc rgb 'orange' title 'unstable: 2 negative', \\
     '' using ($7>=3 ? $3 : 1/0):2 with points pt 7 ps 0.6 lc rgb 'purple' title 'unstable: 3 or more negative'
unset key
set logscale x
set xlabel 'rho_c'
set ylabel 'lowest omega^2'
set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead lc 'black'
plot 'scan_{tag}.dat' using 1:5 with linespoints pt 7 ps 0.4
unset arrow
set ylabel 'number of negative omega^2'
set yrange [-0.2:*]
set ytics 1
plot 'scan_{tag}.dat' using 1:7 with steps lw 2
unset multiplot
"""
        )
    return n, k, recs, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=float, nargs="+", default=[1.0])
    ap.add_argument("--k", type=float, nargs="+", default=[1.0])
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--grade", type=float, default=2.0)
    ap.add_argument("--rho-min", type=float, default=0.01)
    ap.add_argument("--rho-max", type=float, default=1e7)
    ap.add_argument("--ratio", type=float, default=1.25, help="geometric step in rho_c while marching")
    ap.add_argument(
        "--tol", type=float, default=1e-4, help="relative width of the bracket around each stability point"
    )
    ap.add_argument("--band", type=int, default=5, help="extra points on each side of a stability point")
    ap.add_argument("--band-ratio", type=float, default=1.03)
    ap.add_argument(
        "--fill", type=int, default=6, help="extra points inside each interval with an extremum of M"
    )
    ap.add_argument("--out", default=str(SCAN_DIR))
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args()
    jobs = [(n, k, a) for n in a.n for k in a.k]
    results = []
    with Pool(a.procs or min(len(jobs), os.cpu_count())) as pool:
        for n, k, recs, rep in pool.imap_unordered(run_family, jobs):
            print("\n".join(rep), flush=True)
            if recs is not None:
                results.append((n, k, recs))
    if shutil.which("gnuplot"):
        for n, k, _ in results:
            subprocess.run(["gnuplot", f"scan_n{n:g}_k{k:g}.gp"], cwd=a.out, check=False)
            make_svg(a.out, f"scan_n{n:g}_k{k:g}")
        with open(os.path.join(a.out, "scan_overview.gp"), "w") as fh:
            fh.write(
                "set terminal pngcairo size 1000,500\nset output 'scan_overview.png'\n"
                "set grid\nset logscale x\nset xlabel 'rho_c'\nset ylabel 'number of negative omega^2'\n"
                "set ytics 1\nset yrange [-0.2:*]\nset key top left\n"
            )
            fh.write(
                "plot "
                + ", \\\n     ".join(
                    f"'scan_n{n:g}_k{k:g}.dat' using 1:7 with steps lw 2 title 'n={n:g}, k={k:g}'"
                    for n, k, _ in sorted(results, key=lambda q: (q[0], q[1]))
                )
                + "\n"
            )
        subprocess.run(["gnuplot", "scan_overview.gp"], cwd=a.out, check=False)
        make_svg(a.out, "scan_overview")
