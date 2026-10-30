"""Numerical check of the hand-derived perturbation equation against the sympy-linearized
Einstein/conservation equations (res.txt)."""

import random
from sympy import *

t, x = symbols("t x", positive=True)
l, kap = symbols("l kappa", positive=True)
f0 = Function("f0")(x)
g0 = Function("g0")(x)
r0 = Function("rho0")(x)
P0 = Function("P0")(x)
f1 = Function("f1")(t, x)
g1 = Function("g1")(t, x)
r1 = Function("rho1")(t, x)
P1 = Function("P1")(t, x)
zeta = Function("zeta")(t, x)
cs2 = Function("cs2")(x)
d = eval(open("res.txt").read())
res = {k: eval(v) for k, v in d.items()}
E00, E01, E11, D1x = res["('E', (0, 0))"], res["('E', (0, 1))"], res["('E', (1, 1))"], res["('D', 1)"]
s, c = sin(x), cos(x)
L = 1  # l = 1 in kappa
kap_ = kap * l**2  # kappa l^2 appears everywhere

# background rules (l kept inside kap_)
sg = Symbol("gp")
g0p = solve(Eq(kap_ * r0, 2 * c * sg / (g0**3 * s) + 1 / s**2 - 2 / g0**2 - 1 / (g0**2 * s**2) + 2), sg)[0]
a_bg = g0**2 * tan(x) * (1 + kap_ * P0 / 2) + (g0**2 - 1) / (2 * s * c)  # a = f0'/f0 + tan x
f0p = f0 * (a_bg - tan(x))
P0p = -(r0 + P0) * a_bg
r0p = P0p / cs2
rules = {Derivative(g0, x): g0p, Derivative(f0, x): f0p, Derivative(P0, x): P0p, Derivative(r0, x): r0p}


def red(e):
    for _ in range(5):
        reps = {Derivative(b.expr, (x, 2)): diff(rhs, x) for b, rhs in rules.items()}
        e = e.subs(reps).subs(rules).doit()
    return e


h = r0 + P0
# ---- hand-derived pieces
g1h = -(kap_ / 2) * h * g0**3 * tan(x) * zeta
Kf = 3 * tan(x) + 2 / tan(x) + Derivative(g0, x) / g0 - (kap_ / 2) * h * g0**2 * tan(x)
Q = Derivative(zeta, x) + Kf * zeta
r1h = -h * Q - Derivative(r0, x) * zeta
P1h = cs2 * r1h
Theta = cs2 * h
# (f1/f0)' from the xx constraint
fr_p = g0 * g1h * (2 * tan(x) * (1 + kap_ * P0 / 2) + 1 / (s * c)) + (kap_ / 2) * g0**2 * tan(x) * P1h
# Euler:  h g^2 zeta_tt / f0^2 = -P1' - (rho1+P1) a - h*fr_p
ztt = Derivative(zeta, (t, 2))
Euler_mine = h * g0**2 * ztt / f0**2 + Derivative(P1h, x) + (r1h + P1h) * a_bg + h * fr_p  # = 0

# ---- checks 1 & 2: constraints
print("tx:", simplify(E01.subs(g1, g1h).doit()))
# E00 with g1h, r1h should vanish after using background rules
chk00 = E00.subs({g1: g1h, r1: r1h}).doit()
chk11 = E11.subs({g1: g1h, P1: P1h, Derivative(f1, x): f0 * fr_p + f1 * (f0p / f0)}).doit()


def numeric(expr, vals):
    return N(red(expr).subs(vals))


random.seed(3)


def rnd_point():
    return {
        x: random.uniform(0.3, 1.2),
        g0: random.uniform(0.8, 1.6),
        r0: random.uniform(0.05, 1),
        P0: random.uniform(0.01, 0.5),
        f0: random.uniform(0.5, 1.5),
        cs2: random.uniform(0.05, 0.8),
        Derivative(cs2, x): random.uniform(-1, 1),
        kap: random.uniform(1, 30),
        l: 1.0,
    }


def with_zeta(vals, z0, z1, z2, ztt_):
    # replace zeta derivatives by numbers (time dependence enters only via ztt)
    return {**vals}


# unit-vector decomposition in (zeta, zeta_x, zeta_xx, zeta_tt)
Z0, Z1, Z2, Zt = symbols("Z0 Z1 Z2 Zt")


def to_symbols(e):
    e = e.subs(Derivative(zeta, (t, 2)), Zt)
    e = e.subs(Derivative(zeta, (x, 2)), Z2).subs(Derivative(zeta, x), Z1).subs(zeta, Z0)
    return e


def coeffs(expr, vals):
    e = red(expr.doit())
    e = to_symbols(e)
    # remaining derivative of f1 handled by caller
    e = e.subs(vals)
    out = []
    for sym in (Z0, Z1, Z2, Zt):
        other = {y: (1 if y == sym else 0) for y in (Z0, Z1, Z2, Zt)}
        out.append(N(e.subs(other)))
    return out


for i in range(3):
    pt = rnd_point()
    v_ = {**pt}
    a_ = coeffs(chk00, v_)
    print("E00 residual coeffs   :", [f"{float(q):.2e}" for q in a_])
    b_ = coeffs(chk11.subs(f1, 0), v_)
    print("E11 residual coeffs   :", [f"{float(q):.2e}" for q in b_])
    # Euler: D1x with f1' eliminated, f1 kept symbolic (should drop out)
    D = D1x.subs({g1: g1h, r1: r1h, P1: P1h, Derivative(f1, x): f0 * fr_p + f1 * (f0p / f0)}).doit()
    D_c = coeffs(D.subs(f1, 0), v_)
    D_c2 = coeffs(D.subs(f1, 1), v_)
    M_c = coeffs(Euler_mine, v_)
    print("f1-independence of D1x :", [f"{float(p-q):.1e}" for p, q in zip(D_c, D_c2)])
    print(
        "ratio D1x/mine         :",
        [f"{float(p/q):.8g}" if abs(float(q)) > 1e-12 else "-" for p, q in zip(D_c, M_c)],
        flush=True,
    )
