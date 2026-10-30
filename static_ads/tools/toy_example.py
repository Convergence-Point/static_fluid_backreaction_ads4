"""Toy example: the W4 iteration on  x^2 + y^2 = 4,  x^2 y = 1  (Eq. 5 of the W4 paper).

Writes the iteration history to data/toy/x_iterations.dat in a gnuplot-friendly format:
    plot 'data/toy/x_iterations.dat' using 3:4 with linespoints     # path in the (x, y) plane

Usage:  python -m static_ads.tools.toy_example [x0 y0]
"""

import sys

import numpy as np

from ..paths import TOY_DIR
from ..w4 import f_test, save_gnuplot, step


def main():
    x = [float(sys.argv[1]), float(sys.argv[2])] if len(sys.argv) > 2 else [1.0, 0.0]
    delta_tau = 0.5
    p = [0, 0]
    history = [np.asarray(x, dtype=float)]  # x at every iteration, starting with the initial value
    for _ in range(100):
        try:
            x, p = step(x, p, f_test, delta_tau)
        except ZeroDivisionError:  # unpivoted UDL does not exist at this point
            break
        history.append(x)
    print(x, p)
    TOY_DIR.mkdir(parents=True, exist_ok=True)
    save_gnuplot(TOY_DIR / "x_iterations.dat", history, delta_tau)


if __name__ == "__main__":
    main()
