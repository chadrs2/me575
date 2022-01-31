import numpy as np
from math import sqrt

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
