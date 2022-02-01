from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from uncon import uncon, quadratic, rosenbrock, brachistochrone 

if __name__ == '__main__':
    x0 = np.ones((2,1))*50
    tau = 1e-6

    # Global minimum for quadratic should be: x* = [0,0], f(x*) = 0
    xopt, fopt = uncon(quadratic,x0,tau)
    print("SOLUTION:")
    print("X:",xopt)
    print("f:",fopt)