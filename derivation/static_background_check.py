import sympy as sp

t, x, th, ph = sp.symbols("t x theta phi", positive=True)
l, kap = sp.symbols("l kappa", positive=True)
f = sp.Function("f")(x)
g = sp.Function("g")(x)
rho = sp.Function("rho")(x)
P = sp.Function("P")(x)
X = [t, x, th, ph]

W = l**2 / sp.cos(x) ** 2
gm = sp.diag(-W * f**2, W * g**2, W * sp.sin(x) ** 2, W * sp.sin(x) ** 2 * sp.sin(th) ** 2)
gi = gm.inv()
n = 4

Gam = [
    [
        [
            sum(
                gi[a, d] * (sp.diff(gm[d, b], X[c]) + sp.diff(gm[d, c], X[b]) - sp.diff(gm[b, c], X[d]))
                for d in range(n)
            )
            / 2
            for c in range(n)
        ]
        for b in range(n)
    ]
    for a in range(n)
]


def Riem(a, b, c, d):
    r = sp.diff(Gam[a][b][d], X[c]) - sp.diff(Gam[a][b][c], X[d])
    r += sum(Gam[a][c][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][c] for e in range(n))
    return r


Ric = sp.zeros(n)
for b in range(n):
    for d in range(n):
        Ric[b, d] = sp.simplify(sum(Riem(a, b, a, d) for a in range(n)))
R = sp.simplify(sum(gi[a, b] * Ric[a, b] for a in range(n) for b in range(n)))
G = (Ric - R * gm / 2).applyfunc(sp.simplify)

Lam = -3 / l**2
u = [sp.cos(x) / (l * f), 0, 0, 0]
ul = [sum(gm[a, b] * u[b] for b in range(n)) for a in range(n)]
T = sp.zeros(n)
for a in range(n):
    for b in range(n):
        T[a, b] = (rho + P) * ul[a] * ul[b] + P * gm[a, b]

E = (G + Lam * gm - kap * T).applyfunc(sp.simplify)

print("Ric/G diag mixed:")
Gmix = (gi * G).applyfunc(sp.simplify)
for i in range(3):
    print(i, sp.simplify(Gmix[i, i]))
print("R =", R)
print("\nE_tt:", sp.simplify(E[0, 0]))
print("E_xx:", sp.simplify(E[1, 1]))
print("E_thth:", sp.simplify(E[2, 2]))
print("offdiag:", [sp.simplify(E[i, j]) for i in range(n) for j in range(n) if i != j])

# --- solve tt for rho (eq 4), xx for f'/f
Ett = sp.simplify(E[0, 0] / (-W * f**2))  # mixed-like normalisation
Exx = sp.simplify(E[1, 1] / (W * g**2))
fp = sp.Symbol("fp")
gp = sp.Symbol("gp")
gs, rs, Ps, fs = sp.symbols("g_s rho_s P_s f_s")


def sub(e):
    return e.subs(sp.Derivative(f, x), fp).subs(sp.Derivative(g, x), gp).subs({g: gs, rho: rs, P: Ps, f: fs})


Ett_s = sp.simplify(sub(Ett.doit()))
Exx_s = sp.simplify(sub(Exx.doit()))
print("\nEtt (mixed, scalar):", Ett_s)
print("Exx (mixed, scalar):", Exx_s)


import random

c_, s_ = sp.cos(x), sp.sin(x)
eq4 = kap * l**2 * rs - 2 * c_ / (gs**3 * s_) * gp - 1 / s_**2 + 2 / gs**2 + 1 / (gs**2 * s_**2) - 2


def rnd():
    return {
        x: random.uniform(0.2, 1.3),
        gs: random.uniform(0.6, 1.8),
        rs: random.uniform(0, 2),
        Ps: random.uniform(0, 2),
        gp: random.uniform(-2, 2),
        fp: random.uniform(-2, 2),
        fs: random.uniform(0.5, 2),
        kap: random.uniform(0.5, 9),
        l: random.uniform(0.5, 2),
    }


print("tt / eq4 ratio at random points:", [sp.N((Ett_s / eq4).subs(d)) for d in (rnd() for _ in range(4))])
print(
    "tt - eq4/l^2 at random points  :",
    [sp.N((Ett_s - eq4 / l**2).subs(d)) for d in (rnd() for _ in range(4))],
)

sol_fp = sp.solve(Exx_s, fp)[0]
fpf = sol_fp / fs
# conservation (derived from nabla_mu T^{mu x}=0):  P' = -(rho+P)(f'/f + tan x)
P_over = -(fpf + sp.tan(x))
for name, q in (("rho", rs), ("P", Ps)):
    paper = -(gs**2 * sp.tan(x) * (1 + kap * l**2 * q / 2) + (gs**2 - 1) / (2 * s_ * c_))
    print(
        "P'/(rho+P): xx+conservation minus paper eq5 using",
        name,
        ":",
        [sp.N((P_over - paper).subs(d)) for d in (rnd() for _ in range(4))],
    )
# simplified closed form of f'/f + tan x
cand = gs**2 * sp.tan(x) * (1 + kap * l**2 * Ps / 2) + (gs**2 - 1) / (2 * s_ * c_)
print(
    "f'/f + tan x - candidate [P]:",
    [sp.N((fpf + sp.tan(x) - cand).subs(d)) for d in (rnd() for _ in range(4))],
)
print("f'/f from xx, simplified:", sp.simplify(sp.trigsimp(sp.expand_trig(fpf))))
# conservation check
Tup = (gi * T * gi).applyfunc(sp.simplify)
div = sp.simplify(
    sum(sp.diff(Tup[a, 1], X[a]) for a in range(n))
    + sum(Gam[a][a][c] * Tup[c, 1] for a in range(n) for c in range(n))
    + sum(Gam[1][b][c] * Tup[b, c] for b in range(n) for c in range(n))
)
claim = (sp.cos(x) ** 2 * (sp.diff(P, x) + (rho + P) * (sp.diff(f, x) / f + sp.tan(x)))) / (l**2 * g**2)
print("nabla T^{mu x} - claim:", sp.simplify(sp.expand_trig(div - claim)))
