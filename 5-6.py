import numpy as np
import matplotlib.pyplot as plt

from draw import draw_trajectory, draw_conv_factors

def func(x):
    return (10*x[0]**2 - 18*x[0]*x[1] + 10*x[1]**2) / 2 + 4*x[0] - 15*x[1] + 13

def gradient(x):
    return np.array([10*x[0] - 9*x[1] + 4, -9*x[0] + 10*x[1] - 15])

def hessian(x):
    return np.array([[10, -9], [-9, 10]])

def steepestDescent(x, OPTIM=-22):
    states = []

    y = y_prev = func(x)
    while True:
        states.append({
            'x': tuple(x),
            'y': y,
            'conv_factor': (y - OPTIM) / (y_prev - OPTIM)
        })

        g = gradient(x)
        p = -g
        G = hessian(x)
        if np.linalg.norm(g) < 1e-5:
            break

        alpha = - np.dot(p, g) / np.linalg.multi_dot([p, G, p])
        x += alpha * p
        y_prev = y
        y = func(x)

    return states

if __name__ == '__main__':
    x_min = [-1, -1]
    x_max = [12, 7]
    fig, ax = plt.subplots(4,2)
    ax[0][0].set_title('Trajectory')
    ax[0][1].set_title('Convergence Factors')
    for i, x in enumerate([[0,0], [-0.4, 0], [10, 0], [11, 0]]):
        states = steepestDescent(x)
        draw_trajectory(*zip(*[i['x'] for i in states]), x_min, x_max, func, ax[i][0])
        draw_conv_factors([i['conv_factor'] for i in states][1:], ax[i][1])
    plt.show()
