import numpy as np
from math import log
import matplotlib.pyplot as plt

from draw import draw_trajectory

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

def newton(x):
    states = [{'x': x, 'y': func(x)}]
    for i in range(5):
        g = gradient(x)
        G = hessian(x)
        s = - np.dot(np.linalg.inv(G), g)
        x += s

        # import pdb; pdb.set_trace()
        y = func(x)
        # import pdb; pdb.set_trace()
        if np.isnan(y):
            print('!!! out of domain')
            break
        states.append({'x': x, 'y': y})

    return states

if __name__ == '__main__':
    x_min = (1, 20)
    x_max = (15, 90)
    mu = 0.1
    fig, ax = plt.subplots(1, 4)
    for i, x in enumerate([(8, 90), (1, 40), (15, 68.69), (10, 20)]):
        states = newton(x)
        print(states)
        draw_trajectory(*zip(*[i['x'] for i in states]), x_min, x_max, func, ax[i])
    plt.show()
