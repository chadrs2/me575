from threading import local
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from uncon import uncon

def quadratic(x):
    beta = 3./2.
    f = x[0]**2 + x[1]**2 - beta * x[0] * x[1]
    df = np.array([
        2 * x[0] - beta * x[1],
        2 * x[1] - beta * x[0]
        ])    
    return f, df

def quad_helper(x):
    f, df = quadratic(x)
    return f

def rosenbrock(x):
    Sigma = 0
    for i in range(len(x)-1):
        Sigma += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    dSigma = np.zeros(len(x))
    for i in range(len(x)):
        if i == 0:
            dSigma[i] = -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        elif i == len(x)-1:
            dSigma[i] = 200 * (x[i] - x[i-1]**2)
        else:
            dSigma[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
    return Sigma, dSigma

def rosenbrock_helper(x):
    Sigma, dSigma = rosenbrock(x)
    return Sigma

def brachistochrone(yint):
    """ brachistochrone problem.
        
    Parameters
    ----------
    yint : a vector of y location for all the interior points

    Outputs
    -------
    T : scalar proportion to the total time it takes the bead to traverse
        the wire
    dT : dT/dyint the derivatives of T w.r.t. each yint.
    """
     
    # friction
    mu = 0.3
    
    # setup discretization
    h = 1.0
    y = np.concatenate([[h], yint, [0]])
    n = len(y)
    x = np.linspace(0.0, 1.0, n)
    
    # initialize
    T = 0.0
    dT = np.zeros(n-2)
    
    for i in range(n-1):
        ds = sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        vbar = sqrt(h - y[i+1] - mu*x[i+1]) + sqrt(h - y[i] - mu*x[i])
        T += ds/vbar
        
        # gradient
        if i > 0:
            dsdyi = -(y[i+1] - y[i])/ds
            dvdyi = -0.5/sqrt(h - y[i] - mu*x[i])
            dtdyi = (vbar*dsdyi - ds*dvdyi)/(vbar**2)
            dT[i-1] += dtdyi
        
        if i < n-2:
            dsdyip = (y[i+1] - y[i])/ds
            dvdyip = -0.5/sqrt(h - y[i+1] - mu*x[i+1])
            dtdyip = (vbar*dsdyip - ds*dvdyip)/(vbar**2)
            dT[i] += dtdyip
         
    return T, dT

def plot_f(x):
    x1 = np.arange(-1,2,0.01)
    x2 = np.arange(-1,3,0.01)
    X1, X2 = np.meshgrid(x1,x2)
    Z = (1-X1)**2 + 100*(X2-X1**2)**2
    fig,axs = plt.subplots()
    levels = np.arange(0,50,1)
    CS = axs.contour(X1,X2,Z,levels=levels)
    CB = fig.colorbar(CS)
    axs.scatter(x[0],x[1],c='r')
    axs.set_xlabel("X1")
    axs.set_ylabel("X2")
    plt.show()

if __name__ == '__main__':
    tau = 1e-6

    print("*************** QUADRATIC OPTIMIZATION ***************")
    # Global minimum for quadratic should be: x* = [0,0], f(x*) = 0
    x0 = np.ones((2,))
    xopt, fopt = uncon(quadratic,x0,tau)
    print("MY SOLUTION:")
    print("X:",xopt)
    print("f:",fopt)
    print("--------------------")

    X = minimize(quad_helper,x0)
    print("THEIR SOLUTION:")
    print(X)
    
    print()
    print("*************** ROSENBROCK OPTIMIZATION ***************")
    # Global minimum for rosenbrock: x* = [1,...,1], f(x*)=0.0
    # Local minimum for n>=4 rosenbrock: x = [-1,1,...,1]
    x0 = np.zeros((2,))
    xopt, fopt = uncon(rosenbrock,x0,tau)
    # plot_f(xopt)
    print("MY SOLUTION:")
    print("X:",xopt)
    print("f:",fopt)
    print("--------------------")

    X = minimize(rosenbrock_helper,x0)
    print("THEIR SOLUTION:")
    print(X)