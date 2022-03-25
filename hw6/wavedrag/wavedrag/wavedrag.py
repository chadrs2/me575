"""example"""
# r = np.array([0.2170, 0.4343, 0.8110, 1.9433, 3.1663, 4.4278])
# cdw = wavedrag(r)

from json.tool import main
import numpy as np
from math import log, pi
import time
import matplotlib.pyplot as plt

def doFit(x, y, xi, n, ni, ls, rs):

    yi = np.zeros(ni)

    AL = np.zeros((3, 3))
    AR = np.zeros((3, 3))
    BL = np.zeros(3)
    BR = np.zeros(3)
    CL = np.zeros(3)
    CR = np.zeros(3)

    # Parabolic extrapolation
    AL[0, 0] = 1.0
    AL[0, 1] = x[0]
    AL[0, 2] = x[0]**2
    AL[1, 0] = 1.0
    AL[1, 1] = x[1]
    AL[1, 2] = x[1]**2
    AL[2, 0] = 1.0
    AL[2, 1] = x[2]
    AL[2, 2] = x[2]**2

    BL[0] = y[0]
    BL[1] = y[1]
    BL[2] = y[2]

    CL = np.linalg.solve(AL, BL)

    AR[0, 0] = 1
    AR[0, 1] = x[n - 1]
    AR[0, 2] = x[n - 1]**2
    AR[1, 0] = 1
    AR[1, 1] = x[n - 2]
    AR[1, 2] = x[n - 2]**2
    AR[2, 0] = 1
    AR[2, 1] = x[n - 3]
    AR[2, 2] = x[n - 3]**2

    BR[0] = y[n - 1]
    BR[1] = y[n - 2]
    BR[2] = y[n - 3]

    CR = np.linalg.solve(AR, BR)
    
    # Create augmented x and y arrays
    xaug = np.zeros(n + 6)
    xaug[0] = 3 * x[0] - 2 * x[2]
    xaug[1] = 2 * x[0] - x[2]
    xaug[2] = 2 * x[0] - x[1]
    for i in range(0, n):
        xaug[i + 3] = x[i]
    
    xaug[n + 3] = 2 * x[n - 1] - x[n - 2]
    xaug[n + 4] = 2 * x[n - 1] - x[n - 3]
    xaug[n + 5] = 3 * x[n - 1] - 2 * x[n - 3]

    # Create augmented y array
    yaug = np.zeros(n + 6)
    for i in range(0, n):
        yaug[i + 3] = y[i]

    # Will prevent radii from being negative at the left end of the fit
    if (ls == 1):
        if (CL[1] < 0):
            CL[0] = 0.0
            CL[1] = y[1] / x[1]
            CL[2] = 0.0
        
    yaug[0] = CL[0] + CL[1] * xaug[0] + CL[2] * xaug[0]**2
    yaug[1] = CL[0] + CL[1] * xaug[1] + CL[2] * xaug[1]**2
    yaug[2] = CL[0] + CL[1] * xaug[2] + CL[2] * xaug[2]**2


    # Will prevent radii from being negative at the right end of the fit
    if (rs == 1):
        if (2 * CR[2] * x[n - 1] + CR[1] > 0):
            CR[1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
            CR[0] = y[n - 1] - CR[1] * x[n - 1]
            CR[2] = 0

    yaug[n + 3] = CR[0] + CR[1] * xaug[n + 3] + CR[2] * pow(xaug[n + 3], 2)
    yaug[n + 4] = CR[0] + CR[1] * xaug[n + 4] + CR[2] * pow(xaug[n + 4], 2)
    yaug[n + 5] = CR[0] + CR[1] * xaug[n + 5] + CR[2] * pow(xaug[n + 5], 2)

    m = np.zeros(n + 2)
    for i in range(n+2):
        m[i] = (yaug[i + 3] - yaug[i + 2]) / (xaug[i + 3] - xaug[i + 2])
    
    # ================================================================================================#
    # Run fitting algorithm

    t = np.zeros(n)
    f = np.zeros(6)
    v = np.zeros(6)
    d = np.zeros(6)
    w = np.zeros(6)
    fvr = np.zeros(2)

    xim = np.zeros(4)
    yim = np.zeros(4)
    xicll = np.zeros(4)
    yicll = np.zeros(4)
    xicl = np.zeros(4)
    yicl = np.zeros(4)
    xicr = np.zeros(4)
    yicr = np.zeros(4)
    xicrr = np.zeros(4)
    yicrr = np.zeros(4)
    xip = np.zeros(4)
    yip = np.zeros(4)

    for i in range(3, n+3):
        sumw = 0.0
        ii = i - 3
       
        xim[0] = xaug[i]
        xim[1] = xaug[i - 2]
        xim[2] = xaug[i - 1]
        xim[3] = xaug[i - 3]
        yim[0] = yaug[i]
        yim[1] = yaug[i - 2]
        yim[2] = yaug[i - 1]
        yim[3] = yaug[i - 3]
        xicll[0] = xaug[i]
        xicll[1] = xaug[i - 1]
        xicll[2] = xaug[i + 1]
        xicll[3] = xaug[i - 3]
        yicll[0] = yaug[i]
        yicll[1] = yaug[i - 1]
        yicll[2] = yaug[i + 1]
        yicll[3] = yaug[i - 3]
        xicl[0] = xaug[i]
        xicl[1] = xaug[i - 1]
        xicl[2] = xaug[i + 1]
        xicl[3] = xaug[i - 2]
        yicl[0] = yaug[i]
        yicl[1] = yaug[i - 1]
        yicl[2] = yaug[i + 1]
        yicl[3] = yaug[i - 2]
        xicr[0] = xaug[i]
        xicr[1] = xaug[i - 1]
        xicr[2] = xaug[i + 1]
        xicr[3] = xaug[i + 2]
        yicr[0] = yaug[i]
        yicr[1] = yaug[i - 1]
        yicr[2] = yaug[i + 1]
        yicr[3] = yaug[i + 2]
        xicrr[0] = xaug[i]
        xicrr[1] = xaug[i - 1]
        xicrr[2] = xaug[i + 1]
        xicrr[3] = xaug[i + 3]
        yicrr[0] = yaug[i]
        yicrr[1] = yaug[i - 1]
        yicrr[2] = yaug[i + 1]
        yicrr[3] = yaug[i + 3]
        xip[0] = xaug[i]
        xip[1] = xaug[i + 1]
        xip[2] = xaug[i + 2]
        xip[3] = xaug[i + 3]
        yip[0] = yaug[i]
        yip[1] = yaug[i + 1]
        yip[2] = yaug[i + 2]
        yip[3] = yaug[i + 3]

        # Derivative estimates and curvature (volatility measure)
        fvr = Ffunc(xim, yim)
        f[0] = fvr[0]
        v[0] = fvr[1]
        fvr = Ffunc(xicll, yicll)
        f[1] = fvr[0]
        v[1] = fvr[1]
        fvr = Ffunc(xicl, yicl)
        f[2] = fvr[0]
        v[2] = fvr[1]
        fvr = Ffunc(xicr, yicr)
        f[3] = fvr[0]
        v[3] = fvr[1]
        fvr = Ffunc(xicrr, yicrr)
        f[4] = fvr[0]
        v[4] = fvr[1]
        fvr = Ffunc(xip, yip)
        f[5] = fvr[0]
        v[5] = fvr[1]
        
        # Distance measures
        d[0] = Dfunc(xim)
        d[1] = Dfunc(xicll)
        d[2] = Dfunc(xicl)
        d[3] = Dfunc(xicr)
        d[4] = Dfunc(xicrr)
        d[5] = Dfunc(xip)
        # d=[Dfunc(xaug(im)),Dfunc(xaug(icll)),Dfunc(xaug(icl)), ...
        # Dfunc(xaug(icr)),Dfunc(xaug(icrr)),Dfunc(xaug(ip))]
        for j in range(6): 
            v[j] = v[j] + 1.0e-200 # in case some of the v's are zero
            w[j] = 1 / v[j]
        
        
        wmax = np.max(w)
        
        for j in range(6): 
            w[j] = w[j] / (wmax * d[j]) # renormalize to get rid of 1e+200
            t[ii] = t[ii] + f[j] * w[j]
            sumw = sumw + w[j]
        
        t[ii] = t[ii] / sumw
        
    u = np.zeros((4, ni))
    
    for j in range(ni): 
        ii = 1
        while ((xi[j] + 1e-6) > x[ii]):
            ii = ii + 1
            if (xi[j] >= x[n - 1]):
                ii = n - 1
                break

        i = ii - 1
        
        hk = x[i + 1] - x[i]
        p0 = y[i]
        p1 = t[i]
        p2 = (3 * m[ii] - 2 * t[i] - t[i + 1]) / hk
        p3 = (t[i] + t[i + 1] - 2 * m[ii]) / pow(hk, 2)
        
        # Fit value!!!
        hi = xi[j] - x[i]
        yi[j] = p0 + p1 * hi + p2 * hi * hi + p3 * hi * hi * hi
        # Compute the coefficients of the cubic fit
        u[0, j] = p0 - p1 * x[i] + p2 * pow(x[i], 2) - p3 * pow(x[i], 3)
        u[1, j] = p1 - 2 * p2 * x[i] + 3 * p3 * pow(x[i], 2)
        u[2, j] = p2 - 3 * p3 * x[i]
        u[3, j] = p3
        
    return u, yi

    # =============================================================================================#
def Ffunc(x, y): 

    # Initialize needed matrices
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    C = np.zeros((4, 1))
    h = np.zeros(2)

    A[0, 0] = 1
    A[0, 1] = x[0]
    A[0, 2] = pow(x[0], 2)
    A[0, 3] = pow(x[0], 3)
    A[1, 0] = 1
    A[1, 1] = x[1]
    A[1, 2] = pow(x[1], 2)
    A[1, 3] = pow(x[1], 3)
    A[2, 0] = 1
    A[2, 1] = x[2]
    A[2, 2] = pow(x[2], 2)
    A[2, 3] = pow(x[2], 3)
    A[3, 0] = 1
    A[3, 1] = x[3]
    A[3, 2] = pow(x[3], 2)
    A[3, 3] = pow(x[3], 3)

    B[0, 0] = y[0]
    B[1, 0] = y[1]
    B[2, 0] = y[2]
    B[3, 0] = y[3]

    C = np.linalg.solve(A, B)
    
    f = C[1, 0] + 2 * C[2, 0] * x[0] + 3 * C[3, 0] * pow(x[0], 2)
    mx = min(x)
    Mx = max(x)
    h[0] = pow(2 * C[2, 0] + 6 * C[3, 0] * mx, 2)
    h[1] = pow(2 * C[2, 0] + 6 * C[3, 0] * Mx, 2)
    v = max(h)
    fv = np.zeros(2) 
    fv[0] = f
    fv[1] = v
    
    return fv
    

    # =============================================================================================#
def Dfunc(x):

    d = pow(x[1] - x[0], 2) + pow(x[2] - x[0], 2) + pow(x[3] - x[0], 2)

    return d
    


def computeDrag(x, A, nmax):
    rs = 1
    if (abs(A[nmax]) > 1 * pow(10, -10)):
        rs = 0
    
    # Run Akima code to get fit
    u, yi = doFit(x, A, x, nmax + 1, nmax + 1, 1, rs)
    
    # =============================================================================================#
    # Compute double integral
    I = 0
    
    for i in range(nmax):
      
      x1l = x[i]
      x1u = x[i + 1]
      g0 = 2 * u[2][i]
      g1 = 6 * u[3][i]
      dI = 0
      
      for j in range(i+1):
        x2l = x[j]
        x2u = x[j + 1]
        c0 = 2 * u[2][j]
        c1 = 6 * u[3][j]
        
        t1pm = -(c0 / 6) * (3 * g0 + 2 * g1 * (x1u - x2l) + 3 * g1 * x2l) * pow(x1u - x2l, 2)
        t2pm = (c1 / 48) * (4 * g0 + 3 * g1 * (x1u - x2l) + 4 * g1 * x2l) * pow(x1u - x2l, 3)
        t3pm = -(c1 / 12) * (4 * g0 * (x1u - x2l) + 3 * g1 * pow(x1u - x2l, 2) + 6
                    * g0 * x2l + 8 * g1 * (x1u - x2l) * x2l + 6 * g1
                    * pow(x2l, 2)) * pow(x1u - x2l, 2)
        t4pm = (c0 / 36) * (-9 * g0 - 4 * g1 * (x1u - x2l) - 9 * g1 * x2l + 6
                    * (3 * g0 + 2 * g1 * (x1u - x2l) + 3 * g1 * x2l)
                    * log(x1u - x2l)) * pow(x1u - x2l, 2)
        t5pm = -(c1 / 288) * (-16 * g0 - 9 * g1 * (x1u - x2l) - 16 * g1 * x2l + 12
                    * (4 * g0 + 3 * g1 * (x1u - x2l) + 4 * g1 * x2l)
                    * log(x1u - x2l)) * pow(x1u - x2l, 3)
        t6pm = (c1 / 144) * (-4 * g0 * (4 * (x1u - x2l) + 9 * x2l) - g1
                    * (9 * pow(x1u - x2l, 2) + 32 * (x1u - x2l) * x2l + 36 * pow(x2l, 2)) + 12
                    * (4 * g0 * (x1u - x2l) + 3 * g1 * pow(x1u - x2l, 2)
                        + 6 * g0 * x2l + 8 * g1 * (x1u - x2l) * x2l + 6 * g1
                        * pow(x2l, 2)) * log(x1u - x2l)) * pow(x1u - x2l, 2)
        
        if (i == j):
          t1pp = 0
          t1mm = 0
          t1mp = 0
          t2pp = 0
          t2mm = 0
          t2mp = 0
          t3pp = 0
          t3mm = 0
          t3mp = 0
          t4pp = 0
          t4mm = 0
          t4mp = 0
          t5pp = 0
          t5mm = 0
          t5mp = 0
          t6pp = 0
          t6mm = 0
          t6mp = 0
        else:
          
          t1pp = (c0 / 6) * (3 * g0 + 2 * g1 * (x1u - x2u) + 3 * g1 * x2u) * pow(x1u - x2u, 2)
          t1mm = (c0 / 6) * (3 * g0 + 2 * g1 * (x1l - x2l) + 3 * g1 * x2l) * pow(x1l - x2l, 2)
          t2pp = -(c1 / 48) * (4 * g0 + 3 * g1 * (x1u - x2u) + 4 * g1 * x2u) * pow(x1u - x2u, 3)
          t2mm = -(c1 / 48) * (4 * g0 + 3 * g1 * (x1l - x2l) + 4 * g1 * x2l) * pow(x1l - x2l, 3)
          t3pp = (c1 / 12) * (4 * g0 * (x1u - x2u) + 3 * g1 * pow(x1u - x2u, 2) + 6
                      * g0 * x2u + 8 * g1 * (x1u - x2u) * x2u + 6 * g1
                      * pow(x2u, 2)) * pow(x1u - x2u, 2)
          t3mm = (c1 / 12) * (4 * g0 * (x1l - x2l) + 3 * g1 * pow(x1l - x2l, 2) + 6
                      * g0 * x2l + 8 * g1 * (x1l - x2l) * x2l + 6 * g1
                      * pow(x2l, 2)) * pow(x1l - x2l, 2)
          t4pp = -(c0 / 36) * (-9 * g0 - 4 * g1 * (x1u - x2u) - 9 * g1 * x2u + 6
                      * (3 * g0 + 2 * g1 * (x1u - x2u) + 3 * g1 * x2u)
                      * log(x1u - x2u)) * pow(x1u - x2u, 2)
          t4mm = -(c0 / 36) * (-9 * g0 - 4 * g1 * (x1l - x2l) - 9 * g1 * x2l + 6
                      * (3 * g0 + 2 * g1 * (x1l - x2l) + 3 * g1 * x2l)
                      * log(x1l - x2l)) * pow(x1l - x2l, 2)
          t5pp = (c1 / 288) * (-16 * g0 - 9 * g1 * (x1u - x2u) - 16 * g1 * x2u + 12
                      * (4 * g0 + 3 * g1 * (x1u - x2u) + 4 * g1 * x2u)
                      * log(x1u - x2u)) * pow(x1u - x2u, 3)
          t5mm = (c1 / 288) * (-16 * g0 - 9 * g1 * (x1l - x2l) - 16 * g1 * x2l + 12
                      * (4 * g0 + 3 * g1 * (x1l - x2l) + 4 * g1 * x2l)
                      * log(x1l - x2l)) * pow(x1l - x2l, 3)
          t6pp = -(c1 / 144) * (-4 * g0 * (4 * (x1u - x2u) + 9 * x2u) - g1
                      * (9 * pow(x1u - x2u, 2) + 32 * (x1u - x2u) * x2u + 36 * pow(x2u, 2)) + 12
                      * (4 * g0 * (x1u - x2u) + 3 * g1 * pow(x1u - x2u, 2)
                          + 6 * g0 * x2u + 8 * g1 * (x1u - x2u) * x2u + 6 * g1
                          * pow(x2u, 2)) * log(x1u - x2u)) * pow(x1u - x2u, 2)
          t6mm = -(c1 / 144) * (-4 * g0 * (4 * (x1l - x2l) + 9 * x2l) - g1
                      * (9 * pow(x1l - x2l, 2) + 32 * (x1l - x2l) * x2l + 36 * pow(x2l, 2)) + 12
                      * (4 * g0 * (x1l - x2l) + 3 * g1 * pow(x1l - x2l, 2)
                          + 6 * g0 * x2l + 8 * g1 * (x1l - x2l) * x2l + 6 * g1
                          * pow(x2l, 2)) * log(x1l - x2l)) * pow(x1l - x2l, 2)
          
          if (j == i - 1):
            t1mp = 0
            t2mp = 0
            t3mp = 0
            t4mp = 0
            t5mp = 0
            t6mp = 0
          else:
            t1mp = -(c0 / 6) * (3 * g0 + 2 * g1 * (x1l - x2u) + 3 * g1 * x2u) * pow(x1l - x2u, 2)
            t2mp = (c1 / 48) * (4 * g0 + 3 * g1 * (x1l - x2u) + 4 * g1 * x2u) * pow(x1l - x2u, 3)
            t3mp = -(c1 / 12) * (4 * g0 * (x1l - x2u) + 3 * g1 * pow(x1l - x2u, 2)
                        + 6 * g0 * x2u + 8 * g1 * (x1l - x2u) * x2u + 6 * g1
                        * pow(x2u, 2)) * pow(x1l - x2u, 2)
            t4mp = (c0 / 36) * (-9 * g0 - 4 * g1 * (x1l - x2u) - 9 * g1 * x2u + 6
                        * (3 * g0 + 2 * g1 * (x1l - x2u) + 3 * g1 * x2u)
                        * log(x1l - x2u)) * pow(x1l - x2u, 2)
            t5mp = -(c1 / 288) * (-16 * g0 - 9 * g1 * (x1l - x2u) - 16 * g1 * x2u + 12
                        * (4 * g0 + 3 * g1 * (x1l - x2u) + 4 * g1 * x2u)
                        * log(x1l - x2u)) * pow(x1l - x2u, 3)
            t6mp = (c1 / 144) * (-4 * g0 * (4 * (x1l - x2u) + 9 * x2u) - g1
                        * (9 * pow(x1l - x2u, 2) + 32 * (x1l - x2u) * x2u + 36 * pow(x2u, 2)) + 12
                        * (4 * g0 * (x1l - x2u) + 3 * g1
                            * pow(x1l - x2u, 2) + 6 * g0 * x2u + 8 * g1
                            * (x1l - x2u) * x2u + 6 * g1 * pow(x2u, 2))
                        * log(x1l - x2u)) * pow(x1l - x2u, 2)
        
        t1 = t1pp + t1pm + t1mp + t1mm
        t2 = t2pp + t2pm + t2mp + t2mm
        t3 = t3pp + t3pm + t3mp + t3mm
        t4 = t4pp + t4pm + t4mp + t4mm
        t5 = t5pp + t5pm + t5mp + t5mm
        t6 = t6pp + t6pm + t6mp + t6mm
        
        dI = dI + t1 + t2 + t3 + t4 + t5 + t6
      
      I = I + dI
      
    
    # =============================================================================================#
    cdwsref = -2 * I / (2 * pi)
    return cdwsref
  
  

def wavedrag(r):
    # Body parameters
    LENGTH = 100
    MAX_RADIUS = 5
    SREF = pi * MAX_RADIUS * MAX_RADIUS

    # Coarse and fine mesh parameters
    NORMALIZED_X_LOCATIONS = np.array([0, 0.005, 0.01, 0.025, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.975, 0.99, 0.995, 1.0])
    dxTiny = LENGTH / 1e4
    dxFine = LENGTH / 1e2
    dsTiny = LENGTH / 2e2

    finePointCount = (int) ((LENGTH - 2 * dsTiny) / dxFine)
    tinyPointCount = 2 * (int) (dsTiny / dxTiny)
    totalPointCount = finePointCount + tinyPointCount + 1
    # Useful variables to be initialized only once
    X_LOCATIONS = np.zeros(15)
    X_MESH = np.zeros(totalPointCount)
    R_COARSE = np.zeros(15)
    R_FINE = np.zeros(totalPointCount)
    
    # Some radii are just fixed: nose, tail and mid-point
    R_COARSE[0] = 0
    R_COARSE[14] = 0
    R_COARSE[7] = MAX_RADIUS
    # Create absolute x-locations
    for index in range(15): 
        X_LOCATIONS[index] = LENGTH * NORMALIZED_X_LOCATIONS[index]
    index = 0
    # From 0 to dsTiny
    for x in np.arange(0, dsTiny, dxTiny):
        X_MESH[index] = x
        index += 1
    # From dsTiny to LENGTH-dsTiny
    for x in np.arange(dsTiny, LENGTH-dsTiny, dxFine):
        X_MESH[index] = x
        index += 1
    # From LENGTH-dsTiny to LENGTH
    for x in np.arange(LENGTH - dsTiny, LENGTH, dxTiny):
        X_MESH[index] = x
        index += 1
    # Last point
    X_MESH[index] = LENGTH
    
    # Throw exception if dimension mismatch
    if (len(r) != 6): print("Dimension of x MUST be 6"); return 0.0
    # Make full body
    for index in range(6):
        R_COARSE[index + 1] = r[index]
        R_COARSE[13 - index] = r[index]
    
    # Make fine mesh
    u, R_FINE = doFit(X_LOCATIONS, R_COARSE, X_MESH, 15, totalPointCount, 1, 1)
    # Compute areas
    for index in range(totalPointCount): 
        R_FINE[index] = pi * R_FINE[index] * R_FINE[index]
    
    time.sleep(0.1)

    # Code returns cdw*sref, for some reason.
    cdw = computeDrag(X_MESH, R_FINE, totalPointCount - 1) / SREF
    return cdw
    

