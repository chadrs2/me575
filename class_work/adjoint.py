from math import sin, cos
from scipy.optimize import fsolve

def freq(x):

    r = lambda u: u/x + cos(u)

    u0 = 1.0
    u = fsolve(r, u0)[0]

    f = u*x**2

    return f, u


# point at which we will take derivatives
x = 2.0

# compute frequency
f, u = freq(x)

# finite difference
h = 1e-6
fp, _ = freq(x + h)
dfdx_fd = (fp - f)/h

# ---- derive analytic derivatives here ----- 

pfpx = 2*u*x
pfpu = x**2
prpu = 1/x - sin(u)
prpx = -u/x**2

dfdx_an = pfpx - pfpu / prpu * prpx

# ---------------------------------------------

print(dfdx_fd)
print(dfdx_an)