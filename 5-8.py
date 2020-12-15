import numpy as np
from math import log
import matplotlib.pyplot as plt

from draw import draw_trajectory

def in_domain(x):
    return x[0]>0 and x[1]>0 and x[0]+x[1]<100 and x[0]-x[1]<50

def func(x):
    return -9 * x[0] - 10 * x[1] - mu * (np.log(100 - x[0] - x[1]) + np.log(x[0]) + np.log(x[1]) + np.log(50 - x[0] + x[1]))

def gradient(x):
    return np.array([-9 - mu / x[0] + mu / (100-x[0]-x[1]) + mu / (50-x[0]+x[1]),
                     -10 - mu/ x[1] + mu / (100-x[0]-x[1]) - mu / (50-x[0]+x[1])])

def hessian(x):
    return mu * np.array([
            [
                1/x[0]**2 + 1 / (100-x[0]-x[1])**2 + 1 / (50-x[0]+x[1])**2,
                1 / (100-x[0]-x[1])**2 - 1 / (50-x[0]+x[1])**2
            ], [
                1 / (100-x[0]-x[1])**2 - 1 / (50-x[0]+x[1])**2,
                1/x[1]**2 + 1 / (100-x[0]-x[1])**2 + 1 / (50-x[0]+x[1])**2
            ]])

def backtracking_line_search(x, p):
    alpha = 1
    gamma = 0.5
    rho = 1e-4

    g = gradient(x)
    f = func(x)
    # import pdb; pdb.set_trace()
    while not in_domain(x+alpha*p) or func(x+alpha*p) > f + rho*np.dot(g, p)*alpha:
        alpha *= gamma
    return alpha

def newton(x):
    states = [{'x': x, 'y': func(x)}]
    while True:
        g = gradient(x)
        if np.linalg.norm(g) < 1e-5:
            break

        G = hessian(x)
        s = - np.dot(np.linalg.inv(G), g)
        alpha = backtracking_line_search(x, s)
        x += alpha*s

        y = func(x)
        # import pdb; pdb.set_trace()
        if np.isnan(y):
            print('!!! out of domain')
            break
        states.append({'x': tuple(x), 'y': y, 'alpha': alpha})

    return states

if __name__ == '__main__':
    x_min = (0, 0)
    x_max = (80, 100)
    alphas = {}

    fig, ax = plt.subplots(2, 4)

    for idx_mu, mu in enumerate((0.1, 1)):
        for idx, x in enumerate([(8, 90), (1, 40), (15, 68.69), (10, 20)]):
            states = newton(x)
            alphas.setdefault(mu, {})[x] = [s['alpha'] for s in states if 'alpha' in s]
            ax[idx_mu][idx].set_title('initial: {}'.format(x))
            draw_trajectory(*zip(*[i['x'] for i in states]), x_min, x_max, func, ax[idx_mu][idx])

    print(alphas)
    plt.show()
