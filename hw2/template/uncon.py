import numpy as np
from math import sqrt

def quadratic(x):
    beta = 3./2.
    f = x[0]**2 + x[1]**2 - beta * x[0] * x[1]
    df = np.array([
        2 * x[0] - beta * x[1],
        2 * x[1] - beta * x[0]
        ])
    return f, df

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

def uncon(func, x0, tau):
    """An algorithm for unconstrained optimization.

    Parameters
    ----------
    func : function handle
        function handle to a function of the form: f, df = func(x)
        where f is the function value and df is a numpy array containing
        the gradient of f. x are design variables only.
    x0 : ndarray
        starting point
    tau : float
        convergence tolerance.  you should terminate when
        np.max(np.abs(g)) <= tau.  (the infinity norm of the gradient)
    
    Outputs
    -------
    xopt : ndarray
        the optimal solution
    fopt : float
        the corresponding function value
    """

    # Your code goes here!  You can (and should) call other functions, but make
    # sure you do not change the function signature for this file.  This is the
    # file I will call to test your algorithm.
    

    return xopt, fopt
