from __future__ import division
import math
import matplotlib.pyplot as plt
import numpy as np

def _init(x0, t0, tf, tol):
    n = int((tf-t0)/tol)
    t = np.linspace(t0, tf, n+1)
    x = np.zeros(n+1)
    h = np.zeros(n+1)
    x[0] = x0
    h[0] = tol
    return h, t, x,

# Butcher tableau:
# 0   | 0
# 1   | 1   0
# 1/2 | 1/4 1/4 0
# ____|____________
#     | 1/6 1/6 2/3
def rk23(f, x0, t0, tf, tol):
    h, t, x = _init(x0, t0, tf, tol)
    i = 0
    rejected = []

    while t[i] < tf:
        t[i+1] = t[i] + h[i]

        k1 = f(t[i], x[i])
        k2 = f(t[i] + h[i], x[i] + h[i]*k1)
        k3 = f(t[i] + 1/2*h[i], x[i] + 1/4*h[i]*(k1 + k2))

        x2 = x[i] + 1/2*h[i]*(k1 + k2)
        x3 = x[i] + 1/6*h[i]*(k1 + k2 + 4*k3)

        x[i+1] = x2
        lte = x3 - x2
        h_new = h[i]*abs(tol/lte)**(1/3)

        if abs(lte) < 1.1*tol:
            i += 1

        h[i] = h_new

    return t[:i], x[:i], h[:i]


def f(t, x):
    return (1 - 2*t)*x


def main():
    x0 = 1
    t0 = 0
    tf = 4
    tols = [1e-2, 1e-5]

    plt.xlabel("t")
    plt.ylabel("stepsize")
    t, x, h = rk23(f, x0, t0, tf, tols[0])
    plt.plot(t[1:], h[:-1], label="1e-2")
    t, x, h = rk23(f, x0, t0, tf, tols[1])
    plt.plot(t[1:], h[:-1], label="1e-5")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
