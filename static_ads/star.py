"""Static polytropic perfect-fluid star in AdS (Lopes & Okawa).

Profile ODEs (see einstein_derivation.html, section 8), l = 1, P = k rho^(1+1/n):
    g'   = (g^3 tan x / 2) [ kappa rho - csc^2 x + 2/g^2 + 1/(g^2 sin^2 x) - 2 ]
    rho' = -(rho^(1-1/n) (1 + k rho^(1/n)) / (k (1 + 1/n)))
           [ g^2 tan x (1 + kappa P / 2) + (g^2 - 1) / (2 sin x cos x) ]

The code evolves w = rho^(1/n) instead of rho.  Then rho = w^n, P = k w^(n+1) and
    w'   = -(1 + k w) / (k (n + 1)) [ g^2 tan x (1 + kappa P / 2) + (g^2 - 1) / (2 sin x cos x) ]
which has no rho^(1-1/n) factor: w crosses zero linearly at the surface for every n
(rho itself behaves like (x_S - x)^n, which is singular for n < 1).  For n = 1, w = rho.

Boundary conditions: g(0) = 1, rho(x_S) = 0, and ONE of
    rho(0) = rho_c                                   (--rho-c)
    g(x_S) = (1 - 2 M cos^3 x_S / sin x_S)^(-1/2)    (--mass, paper Eq. 7, G = 1)

Solvers:
  Star.shoot()  integrates the IVP from the origin (reference solution)
  solve_w4()    discretizes on x = x_S * s, s in [0, 1], and solves F(u) = 0 with the W4 method

Usage:
  python -m static_ads.star --n 1 --k 1 --rho-c 1 --N 1000
  python -m static_ads.star --n 1.5 --k 1 --mass 1e-3
Each run writes its data to data/runs/<parameters>/ (see run()).
"""

import argparse
import glob
import os
import shutil
import subprocess

import numpy as np
from scipy.integrate import solve_ivp

from .paths import RUNS_DIR
from .w4 import step

KAPPA = 8 * np.pi  # kappa_G l^2 with l = 1  (G = 1)


def mass_from_g(g_s, x_s):
    """Mass from g(x_S) (Eq. 7 inverted), G = 1."""
    return (1 - 1 / g_s**2) * np.sin(x_s) / (2 * np.cos(x_s) ** 3)


class Star:
    """Parameters and discretized equations for one (n, k, boundary condition) problem."""

    def __init__(self, n=1.0, k=1.0, N=1000, rho_c=None, mass=None, grade=1.0):
        if (rho_c is None) == (mass is None):
            raise ValueError("give exactly one of rho_c or mass")
        self.n, self.k, self.N = float(n), float(k), int(N)
        # Graded mesh x = x_S s^grade, s uniform in [0, 1].  grade > 1 clusters nodes near the
        # centre, which is needed when the core is small (large rho_c: core size ~ 1/sqrt(kappa rho_c)).
        self.grade = float(grade)
        self.rho_c, self.mass = rho_c, mass
        self.mass_bc = mass is not None

    # ------------------------------------------------------------------ physics
    def pressure(self, w):
        """P = k rho^(1+1/n) = k w^(n+1)."""
        return self.k * np.abs(w) ** (self.n + 1)

    def density(self, w):
        """rho = |w|^n (abs keeps fractional powers real during iterations)."""
        return np.abs(w) ** self.n

    def rhs(self, x, g, w):
        """Right-hand sides (g', w'), with w = rho^(1/n)."""
        n, k = self.n, self.k
        s, c = np.sin(x), np.cos(x)
        rho = self.density(w)
        P = self.pressure(w)
        dg = 0.5 * g**3 * (s / c) * (KAPPA * rho - 1 / s**2 + 2 / g**2 + 1 / (g**2 * s**2) - 2)
        bracket = g**2 * (s / c) * (1 + KAPPA * P / 2) + (g**2 - 1) / (2 * s * c)
        dw = -(1 + k * w) / (k * (n + 1)) * bracket
        return dg, dw

    def grid(self, x_s):
        """Node positions x_i = x_S (i/N)^grade."""
        return x_s * (np.arange(self.N + 1) / self.N) ** self.grade

    # ----------------------------------------------------------------- shooting
    def shoot(self, rho_c, x0=None):
        """Integrate (g, w) from the origin until w = 0. Returns (x_S, dense solution in (g, w))."""
        if x0 is None:  # start well inside the core, whose size is ~ 1/sqrt(kappa rho_c)
            x0 = min(1e-6, 1e-3 / np.sqrt(KAPPA * max(rho_c, 1.0)))
        y0 = [1 + KAPPA * rho_c * x0**2 / 6, rho_c ** (1 / self.n)]  # series of g near the origin

        def f(x, y):
            return self.rhs(x, y[0], y[1])

        def surface(x, y):
            return y[1]

        surface.terminal = True
        surface.direction = -1

        sol = solve_ivp(
            f, [x0, np.pi / 2 - 1e-3], y0, events=surface, dense_output=True, rtol=1e-12, atol=1e-14
        )
        if sol.t_events[0].size == 0:
            raise RuntimeError("density never reached zero")
        return sol.t_events[0][0], sol

    def shooting_guess(self, rho_c):
        """Shooting solution interpolated on the grid, as a W4 unknown vector."""
        N = self.N
        x_s, sol = self.shoot(rho_c)
        x = self.grid(x_s)
        x[0] = 1e-6
        y = sol.sol(x)
        g, w = y[0].copy(), y[1].copy()
        g[0], w[0], w[-1] = 1.0, rho_c ** (1 / self.n), 0.0
        return pack(g, w, x_s)

    # ------------------------------------------------------------ discretization
    def residual(self, u):
        """Midpoint-rule discretization of the ODEs on x = x_S * i/N, plus the 3 BCs.

        Row order: [g_0 - 1, rho_0 - rho_c, rho_N,   (or [mass condition, g_0 - 1, rho_N, ...]),
                    (g-eq_i, rho-eq_i) for i = 0..N-1].
        The interval equations are multiplied by h = 1/N so all rows are O(1).
        The row and unknown orders matter for the unpivoted UDL (see unpack).
        """
        N = self.N
        g, w, x_s = unpack(np.asarray(u, dtype=float))
        h = 1.0 / N
        sm = (np.arange(N) + 0.5) * h  # midpoints in s (never 0)
        p = self.grade
        xm = x_s * sm**p  # x at the midpoints
        jac = p * sm ** (p - 1)  # dx/ds / x_S
        gm = 0.5 * (g[1:] + g[:-1])
        wm = 0.5 * (w[1:] + w[:-1])
        dg, dw = self.rhs(xm, gm, wm)
        res = np.empty(2 * N + 3)
        if self.mass_bc:
            # Rows 0 and 1 are swapped in this mode: it lowers cond(L), cond(U) from ~1e4..1e5
            # to ~1e2 (found by trying the row orders for which the unpivoted UDL exists).
            arg = np.maximum(1 - 2 * self.mass * np.cos(x_s) ** 3 / np.sin(x_s), 1e-12)
            res[0] = g[-1] - arg**-0.5
            res[1] = g[0] - 1
        else:
            res[0] = g[0] - 1
            res[1] = w[0] - self.rho_c ** (1 / self.n)
        res[2] = w[-1]
        res[3::2] = (g[1:] - g[:-1]) - h * x_s * jac * dg
        res[4::2] = (w[1:] - w[:-1]) - h * x_s * jac * dw
        return res

    def pattern(self):
        """Row indices of the structural non-zeros of each column of dF/du, and a colouring
        of the columns such that columns of one colour never share a row."""
        N = self.N
        n = 2 * N + 3
        rows_of = [None] * n
        rows_of[0] = np.arange(3, n)  # x_S enters every interval equation
        for j in range(N + 1):
            iv = [i for i in (j - 1, j) if 0 <= i < N]
            base = [r for i in iv for r in (3 + 2 * i, 4 + 2 * i)]
            rg, rr = list(base), list(base)
            if j == 0:
                rg.append(1 if self.mass_bc else 0)  # g_0 = 1
                if not self.mass_bc:
                    rr.append(1)  # rho_0 = rho_c
            if j == N:
                rr.append(2)  # rho_N = 0
                if self.mass_bc:
                    rg.append(0)  # mass condition on g_N
            rows_of[1 + 2 * j] = np.array(rg, dtype=int)
            rows_of[2 + 2 * j] = np.array(rr, dtype=int)
        if self.mass_bc:
            rows_of[0] = np.concatenate([[0], rows_of[0]]).astype(int)
        colour = np.empty(n, dtype=int)
        colour[0] = 0
        for j in range(N + 1):
            colour[1 + 2 * j] = 1 + (j % 3)  # g_j
            colour[2 + 2 * j] = 4 + (j % 3)  # rho_j
        groups = [np.flatnonzero(colour == c) for c in range(7)]
        return rows_of, groups


def unpack(u):
    """u = [x_S, g_0, w_0, g_1, w_1, ..., g_N, w_N]   (w = rho^(1/n)).

    x_S comes first on purpose: with x_S last the unpivoted UDL has a zero pivot, and with
    the unknowns grouped as (all g, all rho) it fails too. Putting x_S first gave small
    condition numbers for the L and U factors (cond(L) ~ 60 versus ~1e8 for the
    interleaved order with x_S last), which is what lets W4 converge from poor guesses.
    """
    return u[1::2], u[2::2], u[0]


def pack(g, w, x_s):
    u = np.empty(2 * len(g) + 1)
    u[0] = x_s
    u[1::2] = g
    u[2::2] = w
    return u


def dummy_guess(N, n=1.0, amplitude=1.0, x_s=1.0, grade=1.0):
    """Dummy starting profile: rho(x) = amplitude * (1 - x) on [0, 1], g = 1, x_S = 1
    (stored as w = rho^(1/n))."""
    s = (np.arange(N + 1) / N) ** grade  # = x for x_S = 1
    return pack(np.ones(N + 1), (amplitude * (1 - s)) ** (1 / n), x_s)


# ------------------------------------------------------- sparse W4 step (large N)
#
# Same map as w4.step (paper Eq. 41), but the Jacobian is built with a few
# colour-grouped finite differences and the UDL factors are kept sparse.  The UDL of J is
# computed as an unpivoted sparse LU of the axis-reversed matrix R J R (R = reversal):
#     R J R = Lr Ur,  Ur = Dr Ut   =>   J = (R Lr R)(R Dr R)(R Ut R) = U D L


def jacobian_sparse(F, u, pattern, eps=1e-3):
    """4th-order central-difference Jacobian as a sparse CSC matrix, using the structure above."""
    from scipy.sparse import coo_matrix

    rows_of, groups = pattern
    n = u.size
    ri, ci, vals = [], [], []
    for cols in groups:
        h = eps * np.maximum(1.0, np.abs(u[cols]))

        def shifted(k):
            v = u.copy()
            v[cols] += k * h
            return F(v)

        D = (-shifted(2) + 8 * shifted(1) - 8 * shifted(-1) + shifted(-2)) / 12  # / h per column below
        for c, hc in zip(cols, h):
            r = rows_of[c]
            ri.append(r)
            ci.append(np.full(r.size, c))
            vals.append(D[r] / hc)
    return coo_matrix((np.concatenate(vals), (np.concatenate(ri), np.concatenate(ci))), shape=(n, n)).tocsc()


def udl_sparse(J):
    """Unpivoted sparse UDL of J.  Returns (solve_L, solve_UD) giving L^-1 b and (U D)^-1 b."""
    from scipy.sparse.linalg import splu, spsolve_triangular

    n = J.shape[0]
    Jr = J[::-1, ::-1].tocsc()
    # NATURAL ordering and a zero pivot threshold: use the diagonal whenever it is non-zero.
    try:
        lu = splu(Jr, permc_spec="NATURAL", diag_pivot_thresh=0.0)
    except RuntimeError as e:  # SuperLU: "Factor is exactly singular"
        raise ZeroDivisionError(str(e))
    idx = np.arange(n)
    if not (np.array_equal(lu.perm_r, idx) and np.array_equal(lu.perm_c, idx)):
        raise ZeroDivisionError("zero pivot: unpivoted UDL does not exist here")
    Lr = lu.L.tocsr()  # unit lower triangular
    Ur = lu.U.tocsr()  # upper triangular, diagonal = Dr
    dr = Ur.diagonal()
    # Only an (almost) exactly zero pivot is fatal.  Pivots can span many orders of magnitude on
    # graded meshes (tiny cells near the centre), so an absolute threshold would be wrong.
    if np.any(np.abs(dr) < 1e-300) or not np.all(np.isfinite(dr)):
        raise ZeroDivisionError("zero pivot: unpivoted UDL does not exist here")

    def solve_L(b):  # L^-1 b = R Ur^-1 Dr R b
        return spsolve_triangular(Ur, dr * b[::-1], lower=False)[::-1]

    def solve_UD(b):  # (U D)^-1 b = R Dr^-1 Lr^-1 R b
        return (spsolve_triangular(Lr, b[::-1], lower=True, unit_diagonal=True) / dr)[::-1]

    return solve_L, solve_UD


def step_sparse(u, p, F, pattern, delta_tau=0.5):
    J = jacobian_sparse(F, u, pattern)
    solve_L, solve_UD = udl_sparse(J)
    new_u = u + delta_tau * solve_L(p)
    new_p = (1 - 2 * delta_tau) * p - delta_tau * solve_UD(F(u))
    return new_u, new_p


def solve_w4(star, u0, delta_tau=0.5, tol=1e-12, max_iter=2000, verbose=False, sparse=None):
    """Iterate the W4 map on F(u) = star.residual(u) until max|F| < tol.
    sparse=None picks the sparse step for N > 100. Returns (u, n_iter, converged)."""
    if sparse is None:
        sparse = star.N > 100
    u = np.asarray(u0, dtype=float)
    p = np.zeros_like(u)
    F = star.residual
    pattern = star.pattern() if sparse else None
    for it in range(max_iter):
        err = np.max(np.abs(F(u)))
        if verbose and it % 10 == 0:
            print(f"  iter {it:4d}  max|F| = {err:.3e}")
        # rounding noise in the residual scales with the size of the unknowns (w ~ rho_c can be huge)
        if err < max(tol, 1e-14 * np.max(np.abs(u))):
            return u, it, True
        try:
            if sparse:
                u, p = step_sparse(u, p, F, pattern, delta_tau)
            else:
                u, p = step(u, p, F, delta_tau)
        except (ZeroDivisionError, ValueError, np.linalg.LinAlgError):
            return u, it, False
        if not np.all(np.isfinite(u)):
            return u, it, False
    return u, max_iter, False


def continue_rho(n, k, N, u, rho_from, rho_to, delta_tau=0.5, tol=1e-12, grade=1.0):
    """Move a converged rho_c-mode solution u from rho_from to rho_to.
    Tries the whole jump first; on failure the log-step is halved until it works.
    Returns (u, ok)."""
    frac, cur = 1.0, rho_from
    while abs(np.log(rho_to / cur)) > 1e-12:
        nxt = cur * (rho_to / cur) ** frac
        trial, _, ok = solve_w4(Star(n=n, k=k, N=N, rho_c=nxt, grade=grade), u, delta_tau=delta_tau, tol=tol)
        if ok:
            u, cur = trial, nxt
            frac = min(1.0, 2 * frac)
        else:
            frac /= 2
            if frac < 1e-3:
                return u, False
    return u, True


def solve_dummy(star, amplitude, delta_tau=0.5, tol=1e-12, verbose=False):
    """Solve a rho_c-mode problem starting from the dummy profile rho = amplitude (1 - x).

    The dummy profile has x_S = 1, which can be far from the true radius (e.g. 0.3 for small
    rho_c).  If W4 diverges, retry with a smaller initial x_S and a smaller time step
    (found to be necessary for some (n, k, rho_c); smaller dtau only slows convergence).
    Returns (u, total_iters, ok).
    """
    ladder = [
        (delta_tau, 1.0),
        (delta_tau, 0.5),
        (min(delta_tau, 0.3), 0.5),
        (min(delta_tau, 0.2), 0.3),
        (min(delta_tau, 0.1), 0.1),
    ]
    total = 0
    for dt, x_s0 in ladder:
        u, it, ok = solve_w4(
            star,
            dummy_guess(star.N, star.n, amplitude, x_s0, star.grade),
            delta_tau=dt,
            tol=tol,
            verbose=verbose,
        )
        total += it
        if ok:
            if (dt, x_s0) != ladder[0]:
                print(f"  (needed dtau={dt:g}, initial x_S={x_s0:g})")
            return u, total, True
    return u, total, False


def solve_mass(n, k, N, M_target, rho_start=0.3, delta_tau=0.5, tol=1e-12, verbose=False, grade=1.0):
    """Fix the mass instead of rho_c.

    The mass condition is very nonlinear and W4 does not converge to it from the dummy
    profile.  So: (1) solve with rho(0) = rho_start from the dummy profile rho = rho_start (1 - x);
    (2) continue in M (geometric steps, halved on failure) until M_target.
    Because M(rho_c) has a maximum, the continuation follows the branch that contains
    rho_start; keep rho_start below the maximum-mass density to stay on the stable branch.
    Returns (u, total_iters, ok).
    """
    star = Star(n=n, k=k, N=N, rho_c=rho_start, grade=grade)
    u, it, ok = solve_dummy(star, rho_start, delta_tau=delta_tau, tol=tol)
    if not ok:
        return u, it, False
    g, rho, x_s = unpack(u)
    M_cur = mass_from_g(g[-1], x_s)
    if verbose:
        print(f"  start: rho_c = {rho_start:g} gives M = {M_cur:.4e}")
    frac = 1.0
    total = it
    while abs(np.log(M_target / M_cur)) > 1e-12:
        M_next = M_cur * (M_target / M_cur) ** frac
        trial, it, ok = solve_w4(
            Star(n=n, k=k, N=N, mass=M_next, grade=grade), u, delta_tau=delta_tau, tol=tol
        )
        total += it
        if ok:
            u, M_cur = trial, M_next
            frac = min(1.0, 2 * frac)
            if verbose:
                print(f"  continuation: M = {M_cur:.4e}")
        else:
            frac /= 2
            if frac < 1e-3:
                return u, total, False
    return u, total, True


# ------------------------------------------------------------------ output / driver


def run_dir(star, root=RUNS_DIR):
    bc = f"M{star.mass:g}" if star.mass_bc else f"rhoc{star.rho_c:g}"
    gr = "" if star.grade == 1.0 else f"_p{star.grade:g}"
    return os.path.join(root, f"n{star.n:g}_k{star.k:g}_{bc}_N{star.N}{gr}")


def collect_summary(root=RUNS_DIR):
    """Gather every <root>/*/summary.dat row into <root>/summary.dat (one line per run)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "summary.dat"))):
        with open(f) as fh:
            rows += [ln for ln in fh if not ln.startswith("#")]
    with open(os.path.join(root, "summary.dat"), "w") as fh:
        fh.write("# n k N rho_c mass x_S g_S iters converged\n")
        fh.writelines(rows)


def run(
    n=1.0,
    k=1.0,
    N=1000,
    rho_c=None,
    mass=None,
    delta_tau=0.5,
    tol=1e-12,
    verbose=False,
    plot=True,
    root=RUNS_DIR,
    guess=None,
    rho_start=0.3,
    check_stability=True,
    grade=1.0,
):
    """Solve one star with W4, compare with shooting, and write the results to its folder."""
    star = Star(n=n, k=k, N=N, rho_c=rho_c, mass=mass, grade=grade)
    if star.mass_bc:
        u, n_it, ok = solve_mass(
            star.n,
            star.k,
            N,
            mass,
            rho_start=rho_start,
            delta_tau=delta_tau,
            tol=tol,
            verbose=verbose,
            grade=grade,
        )
    else:
        if guess is None:
            u, n_it, ok = solve_dummy(star, 1.0, delta_tau=delta_tau, tol=tol, verbose=verbose)
        else:
            u, n_it, ok = solve_w4(star, guess, delta_tau=delta_tau, tol=tol, verbose=verbose)
    g, w, x_s = unpack(u)
    rho = star.density(w)
    out = {"star": star, "converged": ok, "iters": n_it, "u": u}
    print(
        f"n={star.n:g} k={star.k:g} "
        + (f"M={star.mass:g}" if star.mass_bc else f"rho_c={star.rho_c:g}")
        + f" N={N}: converged={ok} after {n_it} iterations"
    )
    if not ok:
        return out

    M = mass_from_g(g[-1], x_s)
    rc = rho[0]
    print(f"  W4      : rho_c = {rc:.8g}  x_S = {x_s:.8f}  g_S = {g[-1]:.10f}  M = {M:.8e}")
    # Shooting reference at the central density W4 found
    try:
        xs_sh, sol = star.shoot(rc)
        gs_sh = sol.y[0][-1]
        print(
            f"  shooting: rho_c = {rc:.8g}  x_S = {xs_sh:.8f}  g_S = {gs_sh:.10f}  "
            f"M = {mass_from_g(gs_sh, xs_sh):.8e}"
        )
    except RuntimeError:
        sol, xs_sh = None, None
        print("  shooting: failed (density never reached zero)")

    stable = w2_min = n_neg = None
    if check_stability:
        from .stability import stability

        stable, w2_min, n_neg = stability(star, u)
        print(
            f"  stability: {'STABLE' if stable else 'UNSTABLE'} "
            f"(lowest w^2 = {w2_min:.5g}, negative eigenvalues: {n_neg})"
        )

    d = run_dir(star, root)
    os.makedirs(d, exist_ok=True)
    x = star.grid(x_s)
    cols = [x, g, rho]
    header = "x g_w4 rho_w4"
    if sol is not None:
        y_sh = sol.sol(np.clip(x, 1e-6, xs_sh))
        cols += [y_sh[0], star.density(y_sh[1])]
        header += " g_shoot rho_shoot"
    np.savetxt(os.path.join(d, "star_profile.dat"), np.column_stack(cols), header=header, fmt="%.10g")
    with open(os.path.join(d, "summary.dat"), "w") as fh:
        fh.write("# n k N rho_c mass x_S g_S iters converged\n")
        fh.write(f"{star.n:g} {star.k:g} {N} {rc:.10g} {M:.10g} {x_s:.10g} {g[-1]:.10g} {n_it} {int(ok)}\n")

    title = (
        f"Static polytropic star in AdS (n={star.n:g}, k={star.k:g}, "
        f"rho_c={rc:.4g}, M={M:.4g}, N={N}, x_S={x_s:.4f})"
    )
    overlay_rho = (
        ", \\\n     '' using 1:5 every {e} with points pt 6 title 'shooting'".format(e=max(N // 20, 1))
        if sol is not None
        else ""
    )
    overlay_g = (
        ", \\\n     '' using 1:4 every {e} with points pt 6 title 'shooting'".format(e=max(N // 20, 1))
        if sol is not None
        else ""
    )
    with open(os.path.join(d, "star.gp"), "w") as fh:
        fh.write(
            f"""set terminal pngcairo size 1000,450
set output 'star.png'
set multiplot layout 1,2 title '{title}'
set xlabel 'x'
set xrange [0:{x_s:.6f}]
set grid
set key top right
set ylabel 'rho(x)'
plot 'star_profile.dat' using 1:3 with lines lw 2 title 'W4'{overlay_rho}
set ylabel 'g(x)'
set key top left
plot 'star_profile.dat' using 1:2 with lines lw 2 title 'W4'{overlay_g}
unset multiplot
"""
        )
    if plot and shutil.which("gnuplot"):
        subprocess.run(["gnuplot", "star.gp"], cwd=d, check=False)
    collect_summary(root)
    out.update(rho_c=rc, M=M, x_S=x_s, g_S=g[-1], dir=d, stable=stable, w2_min=w2_min, n_neg=n_neg)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=float, default=1.0, help="polytropic index")
    ap.add_argument("--k", type=float, default=1.0, help="polytropic constant")
    bc = ap.add_mutually_exclusive_group()
    bc.add_argument("--rho-c", type=float, help="central density rho(0) (default 1 if --mass is not given)")
    bc.add_argument("--mass", type=float, help="fix the mass M via g(x_S) (Eq. 7) instead of rho_c")
    ap.add_argument(
        "--rho-start",
        type=float,
        default=0.3,
        help="with --mass: central density of the first solve; sets the branch followed",
    )
    ap.add_argument("--N", type=int, default=1000, help="number of intervals")
    ap.add_argument(
        "--grade",
        type=float,
        default=1.0,
        help="mesh grading exponent, x = x_S s^grade (use 2 for very large rho_c)",
    )
    ap.add_argument("--dtau", type=float, default=0.5, help="W4 time step, 0 < dtau < 1")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--root", default=str(RUNS_DIR), help="output root folder (default: data/runs)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    rho_c = a.rho_c if a.rho_c is not None else (None if a.mass is not None else 1.0)
    run(
        n=a.n,
        k=a.k,
        N=a.N,
        rho_c=rho_c,
        mass=a.mass,
        delta_tau=a.dtau,
        tol=a.tol,
        verbose=a.verbose,
        plot=not a.no_plot,
        root=a.root,
        rho_start=a.rho_start,
        grade=a.grade,
    )
