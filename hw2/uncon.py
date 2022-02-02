import numpy as np
from math import sqrt

def conjugate_grad(df,df_prev,p_prev):
    beta = (df.T @ df) / (df_prev.T @ df_prev)
    p = - df + beta * p_prev
    return p

def backtracking(func, x, alpha0, p):
    alpha = alpha0
    rho = 0.5
    mu1 = 1e-4

    f, df = func(x + alpha * p)
    f0, df0 = func(x)
    phi_alpha = f
    phi_0 = f0
    dphi_0 = df0.T @ p
    
    while phi_alpha > phi_0 + mu1 * alpha * dphi_0:
        alpha = rho * alpha
        
        f, df = func(x + alpha * p)
        phi_alpha = f

    return alpha

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
    x = x0
    # x_normalizer = np.ones_like(x)
    # for i in range(len(x)):
    #     xi_order = len(str(int(x[i]))) - 1
    #     x_normalizer[i] = 1 * 10**(xi_order)
    
    f_prev, df_prev = func(x)
    # f_normalizer = 1.
    # f_order = len(str(int(f_prev))) - 1
    # f_normalizer = 1 * 10**(f_order)
    # print(x,f_prev)
    # print(x_normalizer,f_normalizer)
    # print(df_prev)
    
    p_prev = -df_prev
    alpha = 1.0#0.1
    print("Alpha init:",alpha)
    x = x + alpha * p_prev

    k = 0
    while True:
        f, df = func(x)
        if np.max(np.abs(df)) <= tau:
            break
        p = conjugate_grad(df,df_prev,p_prev)
        alpha = backtracking(func, x, alpha, p)
        
        x = x + alpha * p
        
        f_prev = f
        df_prev = df
        p_prev = p 
        k += 1
    # print("It took k =",k,"operations to converge")
    xopt = x
    fopt = f
    return xopt, fopt
