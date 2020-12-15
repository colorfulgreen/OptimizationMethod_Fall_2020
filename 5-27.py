from sympy import *
import numpy as np

def in_domain(x):
    return x[0]/x[1] < 1/50000

def subs(symbol, x):
    if isinstance(symbol, (np.ndarray, list)):
        return np.array([i.subs({x1: x[0], x2: x[1]}) for i in symbol], dtype=np.float64)
    return symbol.subs({x1: x[0], x2: x[1]})

def residual_symbol():
    c = 96.05
    t = np.array([2000, 5000, 10000, 20000, 30000, 50000])
    d = np.array([0.9427, 0.8616, 0.7384, 0.5362, 0.3739, 0.3096])
    r = (1-x1*t/x2)**(1/(x1*c)-1) - d
    return r

def backtracking_line_search(x, p, f_symbol, g_symbol):
    alpha = 1
    gamma = 0.5
    rho = 1e-4

    g = subs(g_symbol, x)
    f = f_symbol.subs({x1: x[0], x2: x[1]})
    try:
        while not in_domain(x+alpha*p) or (f_symbol.subs({x1: x[0]+alpha*p[0], x2: x[1]+alpha*p[1]}) > f + rho*np.dot(g, p)*alpha):
            alpha *= gamma
    except TypeError:
        import pdb; pdb.set_trace()
    return alpha

def gauss_newton(x, r_symbol):
    # 雅可比
    A_symbol = [diff(r_symbol, x1), diff(r_symbol, x2)]
    # 最小二乘
    f_symbol = 1/2 * np.dot(r_symbol, r_symbol)
    g_symbol = [diff(f_symbol, x1), diff(f_symbol, x2)]

    states = []
    while True:
        print(r_symbol.shape)
        r = subs(r_symbol, x)
        if np.linalg.norm(r) <= 1e-6:
            break
        print(np.linalg.norm(r))
        A_T = np.array([subs(A_symbol[0], x), subs(A_symbol[1], x)], dtype=np.float64)
        try:
            s = -np.linalg.multi_dot([np.linalg.inv(np.dot(A_T, A_T.T)), A_T, r])
        except np.linalg.LinAlgError:
            import pdb; pdb.set_trace()
        alpha = backtracking_line_search(x, s, f_symbol, g_symbol)
        x += alpha * s
    return states

if __name__ == '__main__':
    x1, x2 = symbols('x1 x2', real=True)
    r_symbol = residual_symbol()
    gauss_newton([1, 60000], r_symbol)

