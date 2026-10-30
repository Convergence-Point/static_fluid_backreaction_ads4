"""Figure for the surface boundary behaviour of the radial modes (docs/derivation.html, Sec. 18).

For two stars (n = 1 and n = 3) it shows, as a function of the distance s = x_S - x to the surface:
  (a) the fundamental mode xi(s) (finite) against the excluded local solution ~ s^-n,
  (b) xi, h = rho+P, Theta = c_s^2 h and the Lagrangian pressure change Delta p = -Theta Q on log-log axes,
  (c) the energy accumulated between s_min and s = 0.3 for both solutions (finite vs divergent).

Writes surface_behaviour.png / .svg plus the data files surface_<tag>.dat to data/figures/.
With --publish the two images are also copied to docs/figures/ (the committed copies).

Usage:  python -m static_ads.tools.surface_figure [--publish]
"""

import os
import shutil
import subprocess
import sys
import warnings

import numpy as np

from ..paths import DOCS_FIGURES, FIGURES_DIR
from ..stability import background
from ..star import KAPPA, Star, continue_rho, solve_dummy
from . import build_profiles as B

warnings.filterwarnings("ignore")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    N = 2000
    cases = [(1.0, 0.3), (3.0, 0.5)]
    S_MAX, S_REF = 0.3, 0.2
    gp = [
        "set terminal pngcairo size 1500,900 font 'Helvetica,11'",
        "set output 'surface_behaviour.png'",
        "set multiplot layout 2,3 title 'Radial modes at the stellar surface (s = x_S - x is the distance to the surface)'",
        "set grid",
        "set logscale x",
        "set xrange [*:*] reverse",
        "set xlabel 's = x_S - x   (surface at the right)'",
    ]

    for row, (n, rc) in enumerate(cases):
        st = Star(n=n, k=1, N=N, rho_c=0.01, grade=2.0)
        u, _, ok = solve_dummy(st, 0.01)
        u, ok = continue_rho(n, 1, N, u, 0.01, rc, grade=2.0)
        st = Star(n=n, k=1, N=N, rho_c=rc, grade=2.0)
        vals, vecs = B.modes(st, u)
        xi = vecs[0]
        x, g, w, f = background(st, u)
        mid = lambda q: 0.5 * (q[1:] + q[:-1])
        xm, gm, wm, fm, xim = mid(x), mid(g), mid(w), mid(f), mid(xi)
        rho, P = st.density(wm), st.pressure(wm)
        h = rho + P
        Theta = st.k * (1 + 1 / n) * np.abs(wm) * h
        s_, c_ = np.sin(xm), np.cos(xm)
        gpg = 0.5 * gm**2 * (s_ / c_) * (KAPPA * rho - 1 / s_**2 + 2 / gm**2 + 1 / (gm**2 * s_**2) - 2)
        K = 3 * s_ / c_ + 2 * c_ / s_ + gpg - 0.5 * KAPPA * h * gm**2 * s_ / c_
        E = fm * gm * s_**2 / c_**4
        W = h * gm**2 / fm**2
        dx = np.diff(x)
        xS = x[-1]
        sd = xS - xm  # distance to the surface
        Q = np.diff(xi) / dx + K * xim
        dP = np.abs(Theta * Q)
        # the excluded local solution xi ~ s^-n, matched in size to the physical mode at s = S_REF
        i_ref = np.argmin(np.abs(sd - S_REF))
        xi_sing = xim[i_ref] * (sd / sd[i_ref]) ** (-n)
        Q_sing = n * xi_sing / sd + K * xi_sing  # Q = -d(xi)/ds + K xi
        kin_p, pot_p = E * W * xim**2 * dx, E * Theta * Q**2 * dx
        kin_s, pot_s = E * W * xi_sing**2 * dx, E * Theta * Q_sing**2 * dx
        order = np.argsort(sd)  # ascending distance
        sd, xim, xi_sing, h, Theta, dP, Q = (a[order] for a in (sd, xim, xi_sing, h, Theta, dP, Q))
        kin_p, pot_p, kin_s, pot_s = (a[order] for a in (kin_p, pot_p, kin_s, pot_s))
        keep = sd <= S_MAX

        def tail(a):  # integral from s_min up to S_MAX
            a = np.where(keep, a, 0.0)
            return np.cumsum(a[::-1])[::-1]

        data = np.column_stack(
            [sd, xim, xi_sing, h, Theta, dP, tail(kin_p), tail(pot_p), tail(kin_s), tail(pot_s)]
        )
        data = data[(sd > 5e-4) & keep]
        tag = f"n{n:g}"
        np.savetxt(
            FIGURES_DIR / f"surface_{tag}.dat",
            data,
            header="s xi xi_singular h Theta dP int_kin int_pot int_kin_sing int_pot_sing",
            fmt="%.8g",
        )
        ymax = 4 * np.max(np.abs(xim[(sd > 0.05) & keep]))
        d = f"surface_{tag}.dat"
        ref = data[0]
        gp += [
            f"set title 'n={n:g}, rho_c={rc:g}: displacement of the fundamental mode (omega^2={vals[0]:.3g})'",
            "set logscale x; unset logscale y; set format y '%g'; set ylabel 'xi (normalised)'; set key top left",
            f"set yrange [-0.1:{ymax:.3g}]",
            f"plot '{d}' u 1:2 w l lw 3 lc rgb 'blue' title 'physical mode: finite at the surface', "
            f"'' u 1:3 w l lw 2 dt 2 lc rgb 'red' title 'excluded solution: xi prop. to s^{{-{n:g}}}'",
            f"set title 'n={n:g}: quantities that vanish (or stay finite) at the surface'",
            "set logscale xy; set format y '10^{%L}'; set yrange [*:*]; set ylabel 'value'; set key outside below",
            f"plot '{d}' u 1:(abs($2)) w l lw 2 lc rgb 'blue' title '|xi|', '' u 1:4 w l lw 2 lc rgb 'orange' title 'h = rho+P  (prop. to s^{n:g})', "
            f"'' u 1:5 w l lw 2 lc rgb 'green' title 'Theta = c_s^2 h  (prop. to s^{n + 1:g})', '' u 1:6 w l lw 3 lc rgb 'black' title '|Delta p| = Theta |Q|  (prop. to s^{n + 1:g})', "
            f"{ref[4]:.4g}*(x/{ref[0]:.4g})**{n + 1:g} w l dt 3 lc rgb 'gray' title 'slope s^{n + 1:g}'",
            f"set title 'n={n:g}: energy between s_min and s=0.3'",
            "set logscale xy; set format y '10^{%L}'; set ylabel 'accumulated energy integral'; set key outside below",
            f"plot '{d}' u 1:($7+$8) w l lw 3 lc rgb 'blue' title 'physical mode (kinetic+compressional)', "
            f"'' u 1:($9+$10) w l lw 2 dt 2 lc rgb 'red' title 'excluded solution: xi prop. to s^{{-{n:g}}}'",
        ]
    gp += ["unset multiplot"]
    (FIGURES_DIR / "surface_behaviour.gp").write_text("\n".join(gp) + "\n")
    subprocess.run(["gnuplot", "surface_behaviour.gp"], cwd=FIGURES_DIR, check=False)
    svg = (
        "\n".join(gp)
        .replace(
            "pngcairo size 1500,900 font 'Helvetica,11'",
            "svg size 1500,900 dynamic enhanced font 'Helvetica,11'",
        )
        .replace("surface_behaviour.png", "surface_behaviour.svg")
    )
    (FIGURES_DIR / "surface_behaviour_svg.gp").write_text(svg + "\n")
    subprocess.run(["gnuplot", "surface_behaviour_svg.gp"], cwd=FIGURES_DIR, check=False)
    os.remove(FIGURES_DIR / "surface_behaviour_svg.gp")
    print(f"wrote {FIGURES_DIR}/surface_behaviour.png/.svg")
    if "--publish" in sys.argv:
        DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            shutil.copy(FIGURES_DIR / f"surface_behaviour.{ext}", DOCS_FIGURES / f"surface_behaviour.{ext}")
        print(f"copied to {DOCS_FIGURES}")


if __name__ == "__main__":
    main()
