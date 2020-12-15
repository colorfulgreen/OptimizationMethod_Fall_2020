import matplotlib.pyplot as plt
import numpy as np


def draw_trajectory(x1, x2, x_min, x_max, func, ax, levels=10):
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # contour
    x_steps = np.linspace(x_min, x_max, num=1000)
    x_meshgrid = np.meshgrid(*zip(*x_steps))
    # import pdb; pdb.set_trace()
    y_meshgrid = func(x_meshgrid)
    CS = ax.contour(x_meshgrid[0], x_meshgrid[1], y_meshgrid, levels, extend='both')
    ax.clabel(CS, inline=1, fontsize=10)

    # trajectory
    ax.plot(x1, x2, 'g-*')
    ax.scatter(x1[-1], x2[-1], marker='D', color='', edgecolors='r')


def draw_trajectory_unary(x, y, ax, fmt='g-*'):
    ax.set_title('from {}'.format(x[0]))

    # trajectory
    ax.plot(x, y, fmt)
    ax.scatter(x[-1], y[-1], marker='D', color='', edgecolors='r')


def draw_conv_factors(factors, ax):
    # import pdb; pdb.set_trace()
    ax.set_xlabel('iteration')
    ax.plot(range(len(factors)),
            factors,
            'g-' if len(factors) > 10 else 'g-*')
