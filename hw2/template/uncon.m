function [xopt, fopt] = uncon(func, x0, tau)
%{
An algorithm for unconstrained optimization.

Parameters
----------
func : function handle
    function handle to a function of the form: [f, df] = func(x)
    where f is the function value and df is a *column* vector containing
    the gradient of f.  x are design variables only.
x0 : *column* vector
    starting point
tau : float
    convergence tolerance.  you should terminate when
    norm(df, inf) <= tau.  (the infinity norm of the gradient)

Outputs
-------
xopt : column vector
    the optimal solution
fopt : float
    the corresponding function value
%}

% Your code goes here!  You can (and should) call other functions, but make
% sure you do not change the function signature for this file.  This is the
% file I will call to test your algorithm.  Repeat: don't change the name
% of this file, and don't change the name of this function from uncon.

end
