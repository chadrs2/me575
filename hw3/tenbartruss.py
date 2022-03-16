import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, pi
import cmath
import time

def truss(A):
    """Computes mass and stress for the 10-bar truss problem
    Parameters
    ----------
    A : ndarray of length nbar
        cross-sectional areas of each bar
        see image in book for number order if needed
    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length nbar
        stress in each bar
    """

    # --- specific truss setup -----
    P = 1e5  # applied loads
    Ls = 360.0  # length of sides
    Ld = sqrt(360**2 * 2)  # length of diagonals

    start = [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]
    finish = [3, 1, 4, 2, 3, 1, 4, 3, 2, 1]
    phi = np.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45])*pi/180
    L = np.array([Ls, Ls, Ls, Ls, Ls, Ls, Ld, Ld, Ld, Ld])

    nbar = len(A)  # number of bars
    E = 1e7*np.ones(nbar)  # modulus of elasticity
    rho = 0.1*np.ones(nbar)  # material density

    Fx = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Fy = np.array([0.0, -P, 0.0, -P, 0.0, 0.0])
    rigid = [False, False, False, False, True, True]
    # ------------------

    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom

    # mass
    mass = np.sum(rho*A*L,dtype=complex)

    # stiffness and stress matrices
    K = np.zeros((DOF*n, DOF*n), dtype=complex)
    S = np.zeros((nbar, DOF*n), dtype=complex)

    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        K[np.ix_(idx, idx)] += Ksub
        S[i, idx] = Ssub

    # applied loads
    F = np.zeros((n*DOF, 1))

    for i in range(n):
        idx = node2idx([i+1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]


    # boundary condition
    idx = [i+1 for i, val in enumerate(rigid) if val] # add 1 b.c. made indexing 1-based for convenience
    remove = node2idx(idx, DOF)

    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)

    # solve for deflections
    d = np.linalg.solve(K, F)

    # compute stress
    stress = np.dot(S, d).reshape(nbar)

    return mass, stress, K, S, d, rho, L

def bar(E, A, L, phi):
    """Computes the stiffness and stress matrix for one element
    Parameters
    ----------
    E : float
        modulus of elasticity
    A : float
        cross-sectional area
    L : float
        length of element
    phi : float
        orientation of element
    Outputs
    -------
    K : 4 x 4 ndarray
        stiffness matrix
    S : 1 x 4 ndarray
        stress matrix
    """

    # rename
    c = cos(phi)
    s = sin(phi)

    # stiffness matrix
    k0 = np.array([[c**2, c*s], [c*s, s**2]])
    k1 = np.hstack([k0, -k0])
    K = E*A/L*np.vstack([k1, -k1])

    # stress matrix
    S = E/L*np.array([-c, -s, c, s])

    return K, S

def node2idx(node, DOF):
    """Computes the appropriate indices in the global matrix for
    the corresponding node numbers.  You pass in the number of the node
    (either as a scalar or an array of locations), and the degrees of
    freedom per node and it returns the corresponding indices in
    the global matrices
    """

    idx = np.array([], dtype=int)

    for i in range(len(node)):

        n = node[i]
        start = DOF*(n-1)
        finish = DOF*n

        idx = np.concatenate((idx, np.arange(start, finish, dtype=int)))

    return idx


# FINITE DIFFERENCING - Note: mass, stress, K, S, d, rho, L = truss(x)
def centralDiff_dmdA(x,h):
    dmdA = []
    for i in range(len(x)):
        x_back = x.copy()
        x_back[i] -= h
        x_forward = x.copy()
        x_forward[i] += h
        f_back = truss(x_back)[0]
        f_forward = truss(x_forward)[0]
        dfdx = (f_forward - f_back) / (2*h)
        dmdA.append(np.real(dfdx))
    return np.array(dmdA)

def centralDiff_dsdA(x,h):
    dsdA = []
    for i in range(len(x)):
        x_back = x.copy()
        x_back[i] -= h
        x_forward = x.copy()
        x_forward[i] += h
        g_back = truss(x_back)[1]
        g_forward = truss(x_forward)[1]
        dgdx = (g_forward - g_back) / (2*h)
        dsdA.append(np.real(dgdx))
    return np.array(dsdA)

# COMPLEX STEP - Note: mass, stress, K, S, d, rho, L = truss(x)
def complexStep_dmdA(x,h):
    dmdA = []
    for i in range(len(x)):
        x_imag = np.array(x,dtype=complex)
        x_imag[i] += h*1j
        dfdx_imag = truss(x_imag)[0].imag / h
        dmdA.append(dfdx_imag)
    return np.array(dmdA)

def complexStep_dsdA(x,h):
    dsdA = []
    for i in range(len(x)):
        x_imag = np.array(x,dtype=complex)
        x_imag[i] += h*1j
        dgdx_imag = truss(x_imag)[1].imag / h
        dsdA.append(dgdx_imag)
    return np.array(dsdA)

# IMPLICIT ANALYTIC - Note: mass, stress, K, S, d, rho, L = truss(x)
def implicitAnalytic_dmdA(x):
    rho, L = truss(x)[5:]
    dmdA = rho * L
    return np.array(dmdA)
    
def implicitAnalytic_dsdA(x):    
    m, s, K, S, d, _, _ = truss(x) 
    pspd = np.real(S)
    prpd = np.real(K)
    dsdA = np.zeros((np.size(s,0),np.size(s,0)))
    for i in range(len(x)):
        x_imag = np.array(x,dtype=complex)
        x_imag[i] += 1e-30*1j
        pKpA = truss(x_imag)[2].imag / 1e-30
        prpA = pKpA @ d
        dsdA[i,:] = np.real(- pspd @ np.linalg.inv(prpd) @ prpA).T
    return dsdA


# Optimization Implementation
def runoptimization(stress_maxes, min_areas):
    # Track time lapsed per iteration and how much truss mass has changed
    tstart = time.time()
    t = []
    masses = []

    def objcon(A):
        nonlocal tstart, t, masses
        mass, stress = truss(A)[0:2]
        mass = np.real(mass)
        stress = np.real(stress)
        f = mass
        g = stress_maxes - stress # stress less than max yield stress
        g = np.append(g, stress + stress_maxes) # stress greater than negative max yield stress
        t.append(time.time()-tstart)
        masses.append(f)
        return f, g
    
    Alast = []
    flast = []
    glast = []

    def obj(A):
        nonlocal Alast, flast, glast
        if not np.array_equal(A, Alast):
            flast, glast = objcon(A)
            Alast = A
        return flast

    def con(A):
        nonlocal Alast, flast, glast
        if not np.array_equal(A, Alast):
            flast, glast = objcon(A)
            Alast = A
        return glast

    # Initialize cross-section areas
    # A0 = np.array([8,0.1,8,4,0.1,0.1,6,6,4,0.1])
    A0 = np.ones((10,))*0.5

    # Set bounds and constraints
    lb = min_areas
    ub = np.ones_like(min_areas)*np.inf
    bounds = Bounds(lb,ub,keep_feasible=True)
    constraints = {'type': 'ineq', 'fun': con}
    
    # Run optimization algorithm
    res = minimize(obj, A0, constraints=constraints, bounds=bounds, jac=implicitAnalytic_dmdA)
    print("A =",res.x,"in^2")
    print("f =",res.fun,"lbs")
    print(res.optimality)

    return res.x, res.fun, t, masses

# def plot_convergence(grad_fnorm):
#     x = np.linspace(0,len(grad_fnorm),num=len(grad_fnorm))
#     plt.figure()
#     plt.plot(x,grad_fnorm)
#     plt.xlabel("Iterations")
#     plt.ylabel(r'$||\nabla f||$')
#     plt.yscale("log")
#     plt.show()


if __name__ == '__main__':
    ## Test Derivative Functions ##
    A0 = np.ones((10,))*0.5
    # print("Central Diff - dmdA:",centralDiff_dmdA(A0,1e-8))
    # print("Central Diff - dsdA:",centralDiff_dsdA(A0,1e-8))
    # print("Complex Step - dmdA:",complexStep_dmdA(A0,1e-30))
    # print("Complex Step - dsdA:",complexStep_dsdA(A0,1e-30))
    # print("Implicit Analytic - dmdA:",implicitAnalytic_dmdA(A0))
    # print("Implicit Analytic - dsdA:",implicitAnalytic_dsdA(A0))

    print("Central Difference dmdA Mean Squared Error:",\
        (np.square(implicitAnalytic_dmdA(A0) - centralDiff_dmdA(A0,1e-8))).mean(axis=0))
    print("Central Difference dsdA Mean Squared Error:",\
        (np.square(implicitAnalytic_dsdA(A0) - centralDiff_dsdA(A0,1e-8))).mean(axis=None))
    print("Complex Step Difference dmdA Mean Squared Error:",\
        (np.square(implicitAnalytic_dmdA(A0) - complexStep_dmdA(A0,1e-30))).mean(axis=0))
    print("Complex Step Difference dsdA Mean Squared Error:",\
        (np.square(implicitAnalytic_dsdA(A0) - complexStep_dsdA(A0,1e-30))).mean(axis=None))


    ## Run Optimization ##
    # Set lower bound for cross-sectional area
    min_areas = np.ones((10,))*0.1

    # Set max yield stresses for beams
    stress_maxes = np.ones((10,))*25*10**3
    stress_maxes[8] = 75*10**3

    xstar, fstar, t, masses = runoptimization(stress_maxes, min_areas)

    # Plot change in truss mass over time as optimizer converged
    plt.figure()
    plt.plot(t,masses)
    plt.title("Truss Optimization Results")
    plt.xlabel("t (s)")
    plt.ylabel("Mass (lbs)")
    plt.show()
