import numpy as np
from math import log
import matplotlib.pyplot as plt

from draw import draw_trajectory_unary

def func(x):
    return 9*x - 4*log(x-7)

def gradient(x):
    return 9 - 4 / (x - 7)

def hessian(x):
    return 4 / ((x-7)**2)

def newton(x):
    states = [{'x': x, 'y': func(x)}]
    for i in range(5):
        g = gradient(x)
        G = hessian(x)
        s = -g/G
        x += s

        if x >= 7:
            states.append({'x': x, 'y': func(x)})
        else: # 跳出定义域
            states.append({'x': x, 'y': states[0]['y']})
            break

    return states

if __name__ == '__main__':
    fig, ax = plt.subplots(2, 3)
    for i, x in enumerate((7.4, 7.2, 7.01, 7.8, 7.88, 8)):
        states = newton(x)
        draw_trajectory_unary([i['x'] for i in states],
                              [i['y'] for i in states],
                              ax[i//3][i%3])
    plt.show()

