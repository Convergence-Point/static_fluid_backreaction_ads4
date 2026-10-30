import sympy as sp

t, x, th, ph, eps = sp.symbols("t x theta phi epsilon", positive=True)
l, kap = sp.symbols("l kappa", positive=True)
f0 = sp.Function("f0")(x)
g0 = sp.Function("g0")(x)
r0 = sp.Function("rho0")(x)
P0 = sp.Function("P0")(x)
f1 = sp.Function("f1")(t, x)
g1 = sp.Function("g1")(t, x)
r1 = sp.Function("rho1")(t, x)
P1 = sp.Function("P1")(t, x)
zeta = sp.Function("zeta")(t, x)
F = f0 + eps * f1
G = g0 + eps * g1
rho = r0 + eps * r1
P = P0 + eps * P1
v = eps * sp.diff(zeta, t)
X = [t, x, th, ph]
n = 4
W = l**2 / sp.cos(x) ** 2
gm = sp.diag(-W * F**2, W * G**2, W * sp.sin(x) ** 2, W * sp.sin(x) ** 2 * sp.sin(th) ** 2)
gi = gm.inv()
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
    for d in range(b, n):
        Ric[b, d] = sum(Riem(a, b, a, d) for a in range(n))
        Ric[d, b] = Ric[b, d]
R = sum(gi[a, b] * Ric[a, b] for a in range(n) for b in range(n))
G_ = Ric - R * gm / 2
Lam = -3 / l**2
ut = sp.cos(x) / (l * F)
u = [ut, v * ut, 0, 0]
ul = [sum(gm[a, b] * u[b] for b in range(n)) for a in range(n)]
T = sp.zeros(n)
for a in range(n):
    for b in range(n):
        T[a, b] = (rho + P) * ul[a] * ul[b] + P * gm[a, b]
E = G_ + Lam * gm - kap * T
out = {}
for a, b in [(0, 0), (0, 1), (1, 1), (2, 2)]:
    e = E[a, b]
    e0 = sp.simplify(e.subs(eps, 0))
    e1 = sp.simplify(sp.diff(e, eps).subs(eps, 0))
    out[(a, b)] = (e0, e1)
    print((a, b), "zeroth done", flush=True)
pass
