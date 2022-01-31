"""
    An algorithm for unconstrained optimization.
    
Parameters
----------
func : function handle
    function handle to a function of the form: [f, df] = func(x)
    where f is the function value and df is a vector containing
    the gradient of f.  x are design variables only.
x0 : vector
    starting point
tau : float
    convergence tolerance.  you should terminate when
    norm(df, Inf) <= tau.  (the infinity norm of the gradient)
    
Outputs
-------
xopt : vector
    the optimal solution
fopt : float
    the corresponding function value
"""
function uncon(func, x0, tau)

    # Your code goes here!  You can (and should) call other functions, but make
    # sure you do not change the function signature for this file.  This is the
    # file I will include and call from to test your algorithm.
    
    return xopt, fopt
end
    