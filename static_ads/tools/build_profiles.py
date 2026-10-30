"""Profiles and radial eigenmodes for a selection of stars of each scanned family.

For every family in data/scan/scan_n*_k*.dat a set of central densities is chosen (log-spaced,
plus points on both sides of every stability point), the stars are re-solved with W4 (continuation
along the sequence), and for each star the density profile rho(x), g(x) and the lowest radial
eigenmodes xi_i(x) (eigenvectors of K xi = w^2 Mass xi) are stored in
data/scan/profiles_n<n>_k<k>.json.  interactive_plot reads these files.

Usage:  python -m static_ads.tools.build_profiles [--N 2000] [--per-family 10]
"""

import argparse
import glob
import json
import os
import re
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from ..paths import SCAN_DIR
from ..stability import assemble_banded
from ..star import Star, continue_rho, mass_from_g, solve_dummy, unpack

warnings.filterwarnings("ignore")


def choose(rcs, nneg, per_family):
    idx = {int(np.argmin(np.abs(np.log(rcs / t)))) for t in np.geomspace(rcs[0], rcs[-1], per_family)}
    for i in range(1, len(nneg)):
        if nneg[i] != nneg[i - 1]:
            for j in (i - 6, i - 2, i - 1, i, i + 1, i + 5):
                if 0 <= j < len(rcs):
                    idx.add(j)
    return sorted(idx)


def modes(star, u, n_modes=4):
    dK, eK, dM, eM = assemble_banded(star, u)
    K = diags([eK, dK, eK], [-1, 0, 1], format="csc")
    M = diags([eM, dM, eM], [-1, 0, 1], format="csc")
    vals, vecs = eigsh(K, k=n_modes, M=M, sigma=-20.0, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    out = []
    for j in range(n_modes):
        v = np.concatenate([[0.0], vecs[:, j]])  # xi(0) = 0
        v = v / np.max(np.abs(v))
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        out.append(v)
    return vals, out


def build(args):
    n, k, a = args
    tag = f"n{n:g}_k{k:g}"
    scan = np.loadtxt(os.path.join(a.out, f"scan_{tag}.dat"))
    rcs, nneg = scan[:, 0], scan[:, 6]
    sel = choose(rcs, nneg, a.per_family)
    stars, u, rc_prev = [], None, None
    for i in sel:
        rc = float(rcs[i])
        st = Star(n=n, k=k, N=a.N, rho_c=rc, grade=a.grade)
        if u is None:
            u, _, ok = solve_dummy(Star(n=n, k=k, N=a.N, rho_c=rcs[0], grade=a.grade), rcs[0])
            if not ok:
                return n, k, None
            rc_prev = float(rcs[0])
            if rc != rc_prev:
                u, ok = continue_rho(n, k, a.N, u, rc_prev, rc, grade=a.grade)
                if not ok:
                    continue
        else:
            u2, ok = continue_rho(n, k, a.N, u, rc_prev, rc, grade=a.grade)
            if not ok:
                continue
            u = u2
        rc_prev = rc
        g, w, xs = unpack(u)
        x = st.grid(xs)
        vals, vecs = modes(st, u)
        pick = np.unique(np.round(np.linspace(0, a.N, a.points)).astype(int))
        stars.append(
            dict(
                rc=rc,
                M=float(mass_from_g(g[-1], xs)),
                R=float(xs),
                nneg=int(np.sum(vals < -1e-6)),
                w2=[float(v) for v in vals],
                x=[round(float(v), 7) for v in x[pick]],
                rho=[float(f"{v:.6g}") for v in star_density(st, w)[pick]],
                g=[round(float(v), 7) for v in g[pick]],
                xi=[[round(float(v), 5) for v in m[pick]] for m in vecs],
            )
        )
        print(f"[n={n:g} k={k:g}] rho_c={rc:.4g} done ({len(stars)}/{len(sel)})", flush=True)
    json.dump(stars, open(os.path.join(a.out, f"profiles_{tag}.json"), "w"))
    return n, k, len(stars)


def star_density(st, w):
    return st.density(w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SCAN_DIR))
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--grade", type=float, default=2.0)
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--points", type=int, default=400, help="points kept per stored profile")
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args()
    jobs = []
    for f in sorted(glob.glob(os.path.join(a.out, "scan_n*_k*.dat"))):
        m = re.search(r"scan_n([0-9.]+)_k([0-9.]+)\.dat$", f)
        jobs.append((float(m.group(1)), float(m.group(2)), a))
    with Pool(a.procs or min(len(jobs), os.cpu_count())) as pool:
        for n, k, cnt in pool.imap_unordered(build, jobs):
            print(f"== n={n:g} k={k:g}: {cnt} stars stored", flush=True)
