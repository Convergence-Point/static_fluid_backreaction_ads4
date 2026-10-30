import sympy as sp

exec(open("linearize_base.py").read().split("out = {}")[0])


# first-order pieces
def first(e):
    return sp.diff(e, eps).subs(eps, 0)


E1 = {ab: first(E[ab]) for ab in [(0, 0), (0, 1), (1, 1), (2, 2)]}
E0 = {ab: sp.simplify(E[ab].subs(eps, 0)) for ab in [(0, 0), (0, 1), (1, 1), (2, 2)]}

# --- divergence of T (Euler/continuity) to first order
Tup = gi * T * gi


def div(nu):
    d = sum(sp.diff(Tup[a, nu], X[a]) for a in range(n))
    d += sum(Gam[a][a][c] * Tup[c, nu] for a in range(n) for c in range(n))
    d += sum(Gam[nu][b][c] * Tup[b, c] for b in range(n) for c in range(n))
    return d


D1 = {nu: first(div(nu)) for nu in (0, 1)}
D0 = {nu: sp.simplify(div(nu).subs(eps, 0)) for nu in (0, 1)}
print("D0", D0)

# --- background rules
cs2 = sp.Function("cs2")(x)  # dP0/drho0
s_, c_ = sp.sin(x), sp.cos(x)
g0p = (
    g0**3 * sp.tan(x) / 2 * (kap * l**2 * r0 - 1 / s_**2 + 2 / g0**2 + 1 / (g0**2 * s_**2) - 2) / l**2 * l**2
)
# tt: kappa l^2 rho = 2c g'/(g^3 s) + csc^2 - 2/g^2 - 1/(g^2 s^2) + 2   -> g'
g0p = sp.solve(
    sp.Eq(
        kap * l**2 * r0,
        2 * c_ * sp.Symbol("gp") / (g0**3 * s_) + 1 / s_**2 - 2 / g0**2 - 1 / (g0**2 * s_**2) + 2,
    ),
    sp.Symbol("gp"),
)[0]
f0p_over_f = g0**2 * sp.tan(x) * (1 + kap * l**2 * P0 / 2) + (g0**2 - 1) / (2 * s_ * c_) - sp.tan(x)
f0p = f0 * f0p_over_f
P0p = -(r0 + P0) * (f0p_over_f + sp.tan(x))
r0p = P0p / cs2
rules1 = {
    sp.Derivative(g0, x): g0p,
    sp.Derivative(f0, x): f0p,
    sp.Derivative(P0, x): P0p,
    sp.Derivative(r0, x): r0p,
}


def reduce_bg(e):
    for _ in range(4):
        # second derivatives first
        reps = {}
        for base, rhs in rules1.items():
            f_ = base.expr
            reps[sp.Derivative(f_, (x, 2))] = sp.diff(rhs, x)
        e = e.subs(reps)
        e = e.subs(rules1)
        e = e.doit()
    return e


res = {}
for k_, v_ in E1.items():
    res[("E", k_)] = sp.simplify(reduce_bg(v_))
    print("E1", k_, res[("E", k_)], flush=True)
for k_, v_ in D1.items():
    res[("D", k_)] = sp.simplify(reduce_bg(v_))
    print("D1", k_, res[("D", k_)], flush=True)
open("res.txt", "w").write(repr({str(k): sp.srepr(v) for k, v in res.items()}))
