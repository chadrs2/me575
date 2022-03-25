from me575_ex1 import constraint1
from scipy.optimize import minimize
import numpy as np

def f(x):
    return x[0]**2 + 2 * x[1]**2 + 3 * x[2]**2
def df(x):
    return np.array([2*x[0], 4*x[1], 6*x[2]])


def g1(x):
    return 2*x[0] + x[1] + 2*x[2] - 6
def g1Sigma(x,*args):
    z, sigma = args[0], args[1]
    return 2*x[0] + x[1] + 2*x[2] - 6 - z * sigma
def dg1(x):
    return np.array([2, 1, 2])


def g2(x):
    return 5*x[0] - x[1] - 3*x[2] - 10
def g2Sigma(x,*args):
    z, sigma = args[0], args[1]
    return 5*x[0] - x[1] - 3*x[2] - 10 - z * sigma
def dg2(x):
    return np.array([5, -1, -3])


if __name__ == '__main__':
    x0 = np.array([0,0,0])

    cons = [
        {'type':'ineq','fun':g1},
        {'type':'ineq','fun':g2}
    ]

    x = minimize(f,x0,constraints=cons)
    print("Solution:",x)

    f_star = x.fun
    x_star = x.x

    sigma_x = np.array([0.033, 0.033, 0.0167])

    dG1 = dg1(x_star)
    dG2 = dg2(x_star)

    sigma_g1 = np.sqrt(
        (dG1[0] * sigma_x[0])**2 +
        (dG1[1] * sigma_x[1])**2 +
        (dG1[2] * sigma_x[2])**2
    )
    sigma_g2 = np.sqrt(
        (dG2[0] * sigma_x[0])**2 +
        (dG2[1] * sigma_x[1])**2 +
        (dG2[2] * sigma_x[2])**2
    )
    print(sigma_g1,sigma_g2)

    print()
    z = 3 # CDF of 0.99865
    g1_args = [z,sigma_g1]
    g2_args = [z,sigma_g2]
    cons_sigma = [
        {'type':'ineq','fun':g1Sigma,'args':g1_args},
        {'type':'ineq','fun':g2Sigma,'args':g2_args}
    ]
    x = minimize(f,x0,constraints=cons_sigma)
    print("Forward Propagation Solution:",x)