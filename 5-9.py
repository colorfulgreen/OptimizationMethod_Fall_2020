import numpy as np
from math import log
import matplotlib.pyplot as plt

from draw import draw_trajectory

def func(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def gradient(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2) + 2*x[0] - 2,
                     200*(x[1]-x[0]**2)])

def hessian(x):
    return np.array([
            [
                1200*x[0]**2 - 400*x[1] + 2,
                -400 * x[0]
            ], [
                -400*x[0],
                200
            ]])

def backtracking_line_search(x, p):
    alpha = 1
    gamma = 0.5
    rho = 1e-4

    g = gradient(x)
    f = func(x)
    # import pdb; pdb.set_trace()
    while func(x+alpha*p) > f + rho*np.dot(g, p)*alpha:
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
        print(states[-1])

    return states

def steepest_descent(x):
    states = []

    y = y_prev = func(x)
    while True:
        states.append({
            'x': tuple(x),
            'y': y,
        })
        print(states[-1])

        g = gradient(x)
        p = -g
        G = hessian(x)
        if np.linalg.norm(g) < 1e-5:
            break

        # alpha = - np.dot(p, g) / np.linalg.multi_dot([p, G, p])
        alpha = backtracking_line_search(x, p)
        x += alpha * p
        y_prev = y
        y = func(x)

    return states

if __name__ == '__main__':
    contour_ranges = {
        (1.2, 1.2): ([0.8, 0.8], [1.5, 1.5]),
        (-1.2, 1): ([-1.3,-1.3], [1.2, 1.2])
    }
    contour_levels = [2**i for i in range(-1,10)]

    fig, ax = plt.subplots(2, 2, figsize=(16,8))
    for idx, x in enumerate([(1.2, 1.2), (-1.2, 1)]):
        x_min, x_max = contour_ranges[x]

        states = steepest_descent(x)
        draw_trajectory(*zip(*[i['x'] for i in states]), x_min, x_max, func, ax[idx][0], contour_levels)
        ax[idx][0].set_title('SteepestDescent (iterations:{})'.format(len(states)-1))

        states = newton(x)
        draw_trajectory(*zip(*[i['x'] for i in states]), x_min, x_max, func, ax[idx][1], contour_levels)
        ax[idx][1].set_title('Newton (iterations:{})'.format(len(states)-1))

    plt.show()
