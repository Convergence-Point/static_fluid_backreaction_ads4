"""Radial stability of the static stars (derivation: docs/derivation.html, Part II).

Lagrangian displacement zeta(t, x) = xi(x) e^{i w t}.  With  h = rho + P,  Theta = c_s^2 h (= Gamma P),
W = h g^2 / f^2,  E = f g sin^2 x / cos^4 x,  Q = xi' + K xi  and
K = 3 tan x + 2 cot x + g'/g - (kappa/2) h g^2 tan x, the radial equation is the Sturm-Liouville problem

    (E Theta Q)' - K E Theta Q - E beta xi + w^2 E W xi = 0,

equivalently the variational principle

    w^2 = int E (Theta Q^2 + beta xi^2) dx / int E W xi^2 dx,

with beta(x) given in beta_fn().  Boundary conditions: xi(0) = 0, and the natural condition
Delta p = -Theta Q = 0 at the surface (satisfied automatically since Theta -> 0 there).

Discretisation: linear elements on the star's own grid (coefficients at element midpoints,
consistent mass matrices) give a symmetric generalised eigenproblem  K xi = w^2 Mass xi, with
tridiagonal matrices.  The number of negative eigenvalues is counted exactly with a Sturm sequence,
so the stability test costs O(N).  w^2 < 0 means instability.  (A midpoint-quadrature mass matrix is
singular for alternating modes and gives garbage, hence the consistent one.)

Usage:  python -m static_ads.stability [n] [k]     scans rho_c and writes data/stability/
"""

import numpy as np
from scipy.linalg import eigh

from .paths import STABILITY_DIR
from .star import KAPPA, Star, unpack, mass_from_g, solve_dummy, continue_rho


def beta_fn(x, g, P, h, kap=KAPPA):
    """Potential term beta (gravitational part of the restoring force), T = tan x."""
    T = np.tan(x)
    num = (
        P**2 * T**4 * g**4 * kap**2
        + 6 * P * T**4 * g**4 * kap
        + 2 * P * T**4 * g**2 * kap
        + 2 * P * T**2 * g**4 * kap
        + 2 * P * T**2 * g**2 * kap
        + 9 * T**4 * g**4
        + 6 * T**4 * g**2
        - 7 * T**4
        + 6 * T**2 * g**4
        + 12 * T**2 * g**2
        - 14 * T**2
        + g**4
        + 6 * g**2
        - 7
    )
    return -h * num / (4 * T**2)


def background(star, u, x=None):
    """Background functions on the grid (from the converged W4 unknowns); x defaults to star.grid(x_S)."""
    g, w, x_s = unpack(u)
    if x is None:
        x = star.grid(x_s)
    xs = np.where(x == 0, 1e-12, x)
    s, c = np.sin(xs), np.cos(xs)
    a = g**2 * (s / c) * (1 + KAPPA * star.pressure(w) / 2) + (g**2 - 1) / (2 * s * c)  # f'/f + tan x
    # f(x): f'/f = a - tan x, matched to the exterior (Schwarzschild-AdS) f = 1/g at the surface
    integrand = a - np.tan(xs)
    seg = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x)
    lnf = np.concatenate([[0.0], np.cumsum(seg)])
    lnf = lnf - lnf[-1] + np.log(1.0 / g[-1])
    return x, g, w, np.exp(lnf)


def assemble_banded(star, u, x=None):
    """Tridiagonal matrices of  K xi = w^2 Mass xi  (xi_0 = 0 removed):
    returns (dK, eK, dM, eM) = main and off diagonals of K and Mass, each of length N and N-1."""
    x, g, w, f = background(star, u, x)
    n, k = star.n, star.k
    mid = lambda q: 0.5 * (q[1:] + q[:-1])
    xm, gm, wm, fm = mid(x), mid(g), mid(w), mid(f)  # midpoint values (never at x = 0)
    rho, P = star.density(wm), star.pressure(wm)
    h = rho + P
    cs2 = k * (1 + 1 / n) * np.abs(wm)
    Theta = cs2 * h
    s, c = np.sin(xm), np.cos(xm)
    gp_over_g = 0.5 * gm**2 * (s / c) * (KAPPA * rho - 1 / s**2 + 2 / gm**2 + 1 / (gm**2 * s**2) - 2)
    Kf = 3 * s / c + 2 * c / s + gp_over_g - 0.5 * KAPPA * h * gm**2 * s / c
    E = fm * gm * s**2 / c**4
    W = h * gm**2 / fm**2
    beta = beta_fn(xm, gm, P, h)
    dx = np.diff(x)
    b0, b1 = -1 / dx + Kf / 2, 1 / dx + Kf / 2  # Q = b0 xi_i + b1 xi_{i+1}
    EQ, Eb, EW = E * Theta, E * beta, E * W
    # element matrices (consistent mass for the beta and weight terms: diag 1/3, off-diag 1/6)
    k00 = dx * (EQ * b0 * b0 + Eb / 3)
    k01 = dx * (EQ * b0 * b1 + Eb / 6)
    k11 = dx * (EQ * b1 * b1 + Eb / 3)
    m00, m01 = dx * EW / 3, dx * EW / 6
    dK = np.zeros(star.N + 1)
    dM = np.zeros(star.N + 1)
    dK[:-1] += k00
    dK[1:] += k11
    dM[:-1] += m00
    dM[1:] += m00
    eK, eM = k01, m01
    return dK[1:], eK[1:], dM[1:], eM[1:]  # drop node 0: xi(0) = 0


def assemble(star, u):
    """Dense symmetric (K, Mass) with  K xi = w^2 Mass xi,  i.e. the paper's Eq. (14),
    M xi = w^2 xi with M = Mass^-1 K.  Only sensible for moderate N."""
    dK, eK, dM, eM = assemble_banded(star, u)
    return (np.diag(dK) + np.diag(eK, 1) + np.diag(eK, -1), np.diag(dM) + np.diag(eM, 1) + np.diag(eM, -1))


def spectrum(star, u):
    """All eigenvalues w^2 of the radial perturbation problem, ascending (dense, N <~ 2000)."""
    K, M = assemble(star, u)
    return eigh(K, M, eigvals_only=True)


def _count_below(dK, eK, dM, eM, lam):
    """Number of eigenvalues of the pencil (K, Mass) below lam: the number of negative pivots of
    the tridiagonal LDL^T factorisation of K - lam Mass (Sylvester's law of inertia, Mass > 0)."""
    d = dK - lam * dM
    e = eK - lam * eM
    cnt, piv = 0, d[0]
    if piv == 0.0:
        piv = 1e-300
    cnt += piv < 0
    for i in range(1, d.size):
        piv = d[i] - e[i - 1] ** 2 / piv
        if piv == 0.0:
            piv = 1e-300
        cnt += piv < 0
    return int(cnt)


def lowest_eigenvalues(star, u, n_modes=2, tol=1e-9, x=None):
    """Lowest n_modes eigenvalues by bisection on the Sturm count (O(N) per evaluation)."""
    dK, eK, dM, eM = assemble_banded(star, u, x)
    cnt = lambda lam: _count_below(dK, eK, dM, eM, lam)
    lo = -1.0
    while cnt(lo) > 0:
        lo *= 4
    hi = 1.0
    while cnt(hi) < n_modes:
        hi *= 4
    out = []
    for m in range(1, n_modes + 1):  # smallest lam with cnt(lam) >= m
        a, b = lo, hi
        while b - a > tol * max(1.0, abs(a)):
            mid_ = 0.5 * (a + b)
            if cnt(mid_) >= m:
                b = mid_
            else:
                a = mid_
        out.append(0.5 * (a + b))
    return np.array(out)


def omega2(star, u, n_modes=3):
    """The lowest n_modes eigenvalues w^2."""
    return lowest_eigenvalues(star, u, n_modes)


def stability(star, u, tol=1e-6, x=None):
    """Return (stable, w2_min, n_negative).  The star is unstable if any w^2 < -tol.

    n_negative is the exact number of negative eigenvalues of M (Sylvester inertia of the
    tridiagonal K, since Mass is positive definite); w2_min is the lowest eigenvalue."""
    dK, eK, dM, eM = assemble_banded(star, u, x)
    n_neg = _count_below(dK, eK, dM, eM, -tol)
    w2_min = lowest_eigenvalues(star, u, 1, x=x)[0]
    return n_neg == 0, float(w2_min), int(n_neg)


def family(n=1.0, k=1.0, N=300, rho_values=None):
    """w^2 of the fundamental mode along a sequence of central densities (continuation in rho_c)."""
    rho_values = np.geomspace(0.01, 20, 30) if rho_values is None else rho_values
    out, u, prev = [], None, None
    for rc in rho_values:
        star = Star(n=n, k=k, N=N, rho_c=rc)
        if u is None:
            u, _, ok = solve_dummy(star, rc)
        else:
            u, ok = continue_rho(n, k, N, u, prev, rc)
        if not ok:
            break
        prev = rc
        g, w, x_s = unpack(u)
        ev = spectrum(star, u)
        out.append((rc, mass_from_g(g[-1], x_s), x_s, ev[0], ev[1], ev[2], int(np.sum(ev < -1e-6))))
    return np.array(out)


def main():
    import shutil
    import subprocess
    import sys

    n = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    data = family(n, k)
    print("   rho_c         M         x_S       w^2[0]      w^2[1]      w^2[2]    #negative")
    for row in data:
        print("  ".join(f"{q:10.4e}" for q in row))
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"stability_n{n:g}_k{k:g}"
    np.savetxt(
        STABILITY_DIR / f"{tag}.dat", data, header="rho_c M x_S w2_0 w2_1 w2_2 n_negative", fmt="%.10g"
    )
    script = f"""set terminal pngcairo size 1100,450
set output '{tag}.png'
set multiplot layout 1,2 title 'Radial stability, n={n:g}, k={k:g}'
set grid
set xlabel 'x_S (radius)'
set ylabel 'M'
set key bottom right
plot '{tag}.dat' using 3:($7==0 ? $2 : 1/0) with linespoints pt 7 lc rgb 'blue' title 'stable (all omega^2 > 0)', \\
     '' using 3:($7>0 ? $2 : 1/0) with linespoints pt 7 lc rgb 'red' title 'unstable (some omega^2 < 0)'
set logscale x
set xlabel 'rho_c'
set ylabel 'omega^2'
set arrow from graph 0, first 0 to graph 1, first 0 nohead lc 'black'
set key top right
plot '{tag}.dat' using 1:4 with linespoints title 'lowest eigenvalue', '' using 1:5 with linespoints title 'second lowest'
unset arrow
unset logscale x
unset multiplot
"""
    (STABILITY_DIR / f"{tag}.gp").write_text(script)
    if shutil.which("gnuplot"):
        subprocess.run(["gnuplot", f"{tag}.gp"], cwd=STABILITY_DIR, check=False)


if __name__ == "__main__":
    main()
