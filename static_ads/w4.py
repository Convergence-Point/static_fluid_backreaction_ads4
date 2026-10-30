"""The W4 method for nonlinear systems F(x) = 0 (arXiv:1809.04495).

Damped second-order dynamics, discretised with explicit Euler (paper Eq. 41):

    x_{n+1} = x_n + dtau L^-1 p_n
    p_{n+1} = (1 - 2 dtau) p_n - dtau D^-1 U^-1 F(x_n)

where J(x_n) = U D L is the unpivoted UDL decomposition of the Jacobian.  For 0 < dtau < 1 the
error near a root decays like (1 - dtau)^n.
"""

import numpy as np
from scipy import linalg


def f_test(x):
    return [x[0] * x[0] + x[1] * x[1] - 4, x[0] * x[0] * x[1] - 1]


def jacobian(f, x0, eps=1e-6):
    """Jacobian of f: R^n -> R^m at x0 via 4th-order central finite differences.
    J[i, j] = d f_i / d x_j, using
    (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12 h)."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    m = len(f(x0))
    J = np.empty((m, n))
    for j in range(n):
        # Step scaled to the magnitude of x_j to keep precision reasonable
        h = eps * max(1.0, abs(x0[j]))

        def f_shift(k):
            x = x0.copy()
            x[j] += k * h
            return np.asarray(f(x), dtype=np.float64)

        J[:, j] = (-f_shift(2) + 8 * f_shift(1) - 8 * f_shift(-1) + f_shift(-2)) / (12 * h)
    return J


def step(x, p, f, delta_tau=0.5):
    # Lists don't support arithmetic with floats; work with arrays
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    J = jacobian(f, x)
    UD, L = udl_nopivot(J)  # J = (U D) L (paper, Def. 1)
    # Eq. (41): X = L^-1, Y = D^-1 U^-1 = (U D)^-1
    new_x = x + delta_tau * linalg.solve(L, p)
    new_p = (1 - 2 * delta_tau) * p - delta_tau * linalg.solve(UD, np.asarray(f(x), dtype=float))
    return new_x, new_p


def udl_nopivot(A):
    """Unpivoted UDL decomposition A = U @ diag(D) @ L, with U unit upper triangular,
    D the diagonal (returned as a vector) and L unit lower triangular.
    Gaussian elimination run from the bottom-right corner upwards: at step k the pivot is
    W[k, k], column k above it gives U[:k, k] and row k to its left gives L[k, :k], and
    the Schur complement W[:k, :k] -= outer(U[:k, k], W[k, :k]) is carried to the next step.
    Raises ZeroDivisionError on a zero pivot."""
    W = np.array(A, dtype=float)
    n = W.shape[0]
    U = np.eye(n)
    L = np.eye(n)
    d = np.empty(n)
    for k in range(n - 1, -1, -1):
        if abs(W[k, k]) < 1e-14:
            raise ZeroDivisionError(f"Zero pivot at index {k}; unpivoted UDL doesn't exist here.")
        d[k] = W[k, k]
        U[:k, k] = W[:k, k] / d[k]
        L[k, :k] = W[k, :k] / d[k]
        W[:k, :k] -= d[k] * np.outer(U[:k, k], L[k, :k])
    return U * d, L  # U * d scales column j of U by d[j], i.e. U @ diag(d)


def save_gnuplot(path, history, delta_tau):
    """Write whitespace-separated columns: iter, tau, x0, x1, ...
    gnuplot examples:
      plot 'history.dat' using 2:3 with lines title 'x0', '' using 2:4 with lines title 'x1'
      plot 'history.dat' using 3:4 with linespoints   # trajectory in the (x0, x1) plane
    """
    xs = np.array(history)
    iters = np.arange(len(xs))
    data = np.column_stack([iters, iters * delta_tau, xs])
    header = "iter tau " + " ".join(f"x{j}" for j in range(xs.shape[1]))
    np.savetxt(path, data, header=header, fmt="%.10g")


def lu_nopivot(A):
    """Unpivoted Doolittle LU: A = L @ U, L unit lower, U general upper.
    Raises if a zero pivot is hit (no pivoting available to fix it)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        if abs(U[k, k]) < 1e-14:
            raise ZeroDivisionError(f"Zero pivot at step {k}; unpivoted UDL doesn't exist here.")
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return L, U
