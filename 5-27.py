from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def in_domain(x):
    return x[0] < 1/50000

def subs(symbol, x):
    if isinstance(symbol, (np.ndarray, list)):
        return np.array([i.subs({x1: x[0], x2: x[1]}) for i in symbol], dtype=np.float64)
    return symbol.subs({x1: x[0], x2: x[1]})

def backtracking_line_search(x, p, f_symbol, g_symbol):
    alpha = 1
    gamma = 0.6
    rho = 0.5

    g = subs(g_symbol, x)
    f = f_symbol.subs({x1: x[0], x2: x[1]})
    while not in_domain(x+alpha*p) or (f_symbol.subs(zip([x1,x2], x+alpha*p)) > f + rho*np.dot(g, p)*alpha):
        alpha *= gamma
    return alpha

def gauss_newton(x, r_symbol):
    # 雅可比
    A_symbol = [diff(r_symbol, x1), diff(r_symbol, x2)]
    # 最小二乘
    f_symbol = 1/2 * np.dot(r_symbol, r_symbol)
    g_symbol = [diff(f_symbol, x1), diff(f_symbol, x2)]

    states = []
    idx = 0
    while True:
        #import pdb; pdb.set_trace()
        r = subs(r_symbol, x)
        states.append({'x': tuple(x), 'r': np.linalg.norm(r)})
        if np.linalg.norm(r) <= 0.08:
            break
        A_T = np.array([subs(A_symbol[0], x), subs(A_symbol[1], x)], dtype=np.float64)
        try:
            s = -np.linalg.multi_dot([np.linalg.inv(np.dot(A_T, A_T.T)), A_T, r])
        except np.linalg.LinAlgError:
            pass
        alpha = backtracking_line_search(x, s, f_symbol, g_symbol)
        x += alpha * s
        idx += 1
        print(r, np.linalg.norm(r))
        print(x, idx)
    return states

if __name__ == '__main__':
    x1, x2 = symbols('x1 x2', real=True)
    c = 96.05
    t = np.array([2000, 5000, 10000, 20000, 30000, 50000])
    d = np.array([0.9427, 0.8616, 0.7384, 0.5362, 0.3739, 0.3096])
    phi_symbol = (1-x1*t)**(x2-1)
    r_symbol = phi_symbol - d

    fig, ax = plt.subplots(2,2,figsize=(16,8))
    for idx, x in enumerate([(-1, -2)]):
        print('===========', idx, x)
        states = gauss_newton(x, r_symbol)
        print(states)

        ax[idx][0].plot(range(len(states)), [i['r'] for i in states], 'g-*')
        ax[idx][0].set_xlabel('iteration')
        ax[idx][0].set_ylabel('2-norm of residual')

        x1_value, x2_value = states[-1]['x']
        t_values = range(2000, 50000, 2)
        d_values = [(1-x1_value*t_value)**(x2_value-1) for t_value in t_values]
        ax[idx][1].plot(t_values, d_values)
        ax[idx][1].scatter(t, d)
        ax[idx][1].set_xlabel('t')
        ax[idx][1].set_ylabel('d')

    plt.show()
