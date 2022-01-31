function [T, dT] = brachistochrone(yint)
%{
brachistochrone problem.

Parameters
----------
yint : a vector of y location for all the interior points

Outputs
-------
T : scalar proportion to the total time it takes the bead to traverse
    the wire
dT : dT/dyint the derivatives of T w.r.t. each yint.
%}

% friction
mu = 0.3;

% check input
if ~iscolumn(yint)
    yint = yint';
end

% setup discretization
h = 1;
y = [h; yint; 0];
n = length(y);
x = linspace(0.0, 1.0, n);

% initialize
T = 0.0;
dT = zeros(n-2, 1);

for i = 1:n-1 
    ds = sqrt((x(i+1) - x(i))^2 + (y(i+1) - y(i))^2);
    vbar = sqrt(h - y(i+1) - mu*x(i+1)) + sqrt(h - y(i) - mu*x(i));
    T = T + ds/vbar;
    
    % gradient
    if i > 1
        dsdyi = -(y(i+1) - y(i))/ds;
        dvdyi = -0.5/sqrt(h - y(i) - mu*x(i));
        dtdyi = (vbar*dsdyi - ds*dvdyi)/(vbar^2);
        dT(i-1) = dT(i-1) + dtdyi;
    end
    if i < n-1
        dsdyip = (y(i+1) - y(i))/ds;
        dvdyip = -0.5/sqrt(h - y(i+1) - mu*x(i+1));
        dtdyip = (vbar*dsdyip - ds*dvdyip)/(vbar^2);
        dT(i) = dT(i) + dtdyip;
    end
     
end


end