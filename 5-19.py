import numpy as np
from math import log
import matplotlib.pyplot as plt

from draw import draw_trajectory

def hilbert_matrix(n):
    return np.array([[1/(i+j-1) for j in range(1,n+1)] for i in range(1,n+1)])

def conjugate_gradient(x, G, b):
    states = []
    g = np.dot(G,x) - b
    p = -g
    while np.linalg.norm(g) > 1e-5:
        #import pdb; pdb.set_trace()
        d = np.dot(G, p)
        alpha = np.dot(g, g) / np.dot(p, d)
        x += alpha * p
        g_prev = g.copy()
        g += alpha * d
        beta = np.dot(g, g) / np.dot(g_prev, g_prev)
        p = -g + beta * p
        states.append({'x': x, 'g': g.copy(), 'alpha': alpha, 'beta': beta})
    return states

if __name__ == '__main__':
    for n in [5, 8, 12, 20]:
        print('n={}'.format(n))
        G = hilbert_matrix(n)
        b = np.ones(n)
        x = np.zeros(n)
        states = conjugate_gradient(x, G, b)
        print(len(states))

'''
n=5
6
n=8
19
n=12
38
n=20
75
'''

