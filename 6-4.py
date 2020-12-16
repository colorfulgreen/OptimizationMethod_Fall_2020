import numpy as np
import matplotlib.pyplot as plt
from math import log

def func(x, n):
    return sum(10*(x[2*i+1]-x[2*i]**2)**2 + (1-x[2*i])**2 for i in range(n))

def gradient(x, n):
    g = []
    for i in range(n):
        g.extend([-40*x[2*i]*(x[2*i+1]-x[2*i]**2) + 2*x[2*i] - 2,
                  20*(x[2*i+1]-x[2*i]**2)])
    return np.array(g)

def hessian(x, n):
    G = np.zeros((2*n, 2*n))
    for i in range(n):
        G[2*i:2*(i+1), 2*i:2*(i+1)] = np.array([
                [
                    120*x[2*i]**2 - 40*x[2*i+1] + 2,
                    -40 * x[2*i]
                ], [
                    -40*x[2*i],
                    20
                ]])
    return G


def find_tau(xj, pj, delta):
    pjx = np.dot(pj, xj)
    pjp = np.dot(pj, pj)
    xjx = np.dot(xj, xj)
    res1 = (-2 * (pjx) + np.sqrt((2 * pjx)**2 - 4 * pjp *
                                 (xjx - delta**2))) / (2 * pjp)
    res2 = (-2 * (pjx) - np.sqrt((2 * pjx)**2 - 4 * pjp *
                                 (xjx - delta**2))) / (2 * pjp)
    if res1 >= 0 and res2 < 0:
        return res1
    elif res2 >= 0 and res1 < 0:
        return res2
    elif res1 >= 0 and res2 >= 0:
        return False
    else:
        return False

def steihaug_conjugate_gradient(g, B, delta, n):
    x = np.zeros(2 * n)
    r0 = g
    r = r0
    p = -g

    while True:
        if np.linalg.multi_dot([p, B, p]) <= 0:
            print("遇到非正曲率")
            tau = find_tau(x, p, delta)
            return x + tau * p

        alpha = np.dot(r, r) / np.linalg.multi_dot([p, B, p])
        x_next = x + alpha * p
        r_next = r + alpha * np.dot(B, p)

        if np.linalg.norm(x_next, 2) >= delta:
            tau = find_tau(x, p, delta)
            print("达到信赖域的边界")
            return x + tau * p

        if np.linalg.norm(r_next) < 1e-6 * np.linalg.norm(r0):
            print("满足停止测试")
            return x_next

        beta = np.dot(r_next, r_next) / np.dot(r, r)
        p = -r_next + beta * p
        r = r_next
        x = x_next

def trust_region(x, n):
    delta=1
    gamma_d=0.5

    states = []
    while True:
        states.append({'x': x, 'f': func(x, n)})
        g = gradient(x, n)
        if np.linalg.norm(g) < 1e-5:
            break
        G = hessian(x, n)
        s = steihaug_conjugate_gradient(g, G, delta, n)

        rho = -(func(x, n) - func(x+s, n)) / \
              (np.dot(s, g) + 1/2 * np.linalg.multi_dot([s, G, s]))

        if rho < 0.25:
            delta = np.linalg.norm(s) / 4
        if rho >= 0.75 and np.linalg.norm(s) == delta:
            delta *= 2

        if rho > 0:
            x += s

    return states

if __name__ == "__main__":
    fig, ax = plt.subplots(1,2,figsize=(16,8))
    for idx, n in enumerate([10, 50]):
        x = np.ones(2*n) * 5
        states = trust_region(x, n)

        ax[idx].plot(range(len(states)), [s['f'] for s in states], 'g-*')
        ax[idx].set_xlabel('iteration')
        ax[idx].set_ylabel('f(x)')
        ax[idx].set_title('n={}, initial={}'.format(n, 5))
    plt.show()
