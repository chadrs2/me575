import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(grad_fnorm):
    x = np.linspace(0,len(grad_fnorm),num=len(grad_fnorm))
    plt.figure()
    plt.plot(x,grad_fnorm)
    plt.xlabel("Iterations")
    plt.ylabel(r'$||\nabla f||$')
    plt.yscale("log")
    plt.show()

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
    x = x0
    
    f, df = func(x)
    alpha_init = 0.01

    k = 0
    grad_fnorm = []
    while True:
        grad_fnorm.append(np.max(np.abs(df)))
        if np.max(np.abs(df)) <= tau:
            break

        # Choose search direction
        if k == 0:
            p = -df / np.linalg.norm(df)
            # # Choose step size
            # alpha = backtracking(func, x, alpha_init, p)
        else:
            f, df = func(x)
            p = conjugate_grad(df,df_prev,p)
            # # Choose step size
            # alpha = backtracking(func, x, alpha, p)

        # Choose step size
        alpha = backtracking(func, x, alpha_init, p)

        # Step along conjugate gradient
        x = x + alpha * p
        
        # Update values
        df_prev = df
        k += 1
    
    plot_convergence(grad_fnorm)
    xopt = x
    fopt = f
    return xopt, fopt
