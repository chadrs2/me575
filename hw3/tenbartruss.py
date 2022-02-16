import numpy as np
from math import sin, cos, sqrt, pi
import cmath

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

    return mass, stress, K, S, d



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

# Derivative helper functions:
def getMass(A):
    mass, stress, K, S, d = truss(A)
    return mass

def getStress(A):
    mass, stress, K, S, d = truss(A)
    return stress

def getK(A):
    mass, stress, K, S, d = truss(A)
    return K

# FINITE DIFFERENCING
def centralDiff_dmdA(x,h):
    dmdA = []
    for i in range(len(x)):
        x_back = x.copy()
        x_back[i] -= h
        x_forward = x.copy()
        x_forward[i] += h
        f_back = getMass(x_back)
        f_forward = getMass(x_forward)
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
        g_back = getStress(x_back)
        g_forward = getStress(x_forward)
        dgdx = (g_forward - g_back) / (2*h)
        dsdA.append(np.real(dgdx))
    return np.array(dsdA)

# COMPLEX STEP
def complexStep_dmdA(x,h):
    dmdA = []
    for i in range(len(x)):
        x_imag = np.array(x,dtype=complex)
        x_imag[i] += h*1j
        dfdx_imag = getMass(x_imag).imag / h
        dmdA.append(dfdx_imag)
    return np.array(dmdA)

def complexStep_dsdA(x,h):
    dsdA = []
    for i in range(len(x)):
        x_imag = np.array(x,dtype=complex)
        x_imag[i] += h*1j
        dgdx_imag = getStress(x_imag).imag / h
        dsdA.append(dgdx_imag)
    return np.array(dsdA)

# IMPLICIT ANALYTIC
# def implicitAnalytic_dmdA(func,x):
    

def implicitAnalytic_dsdA(func,x):
    m, s, K, S, d = func(x) 
    pspd = np.real(S)
    pspA = np.zeros_like(S)
    prpd = -np.real(K)
    pKpA = getK(x+1e-30*1j).imag / 1e-30
    prpA = - pKpA * d

    return pspA - pspd @ np.linalg.inv(prpd) @ prpA


if __name__ == '__main__':
    A0 = np.ones((10,))*0.5
    print("Central Diff - dmdA:",centralDiff_dmdA(A0,1e-8))
    print("Central Diff - dsdA:",centralDiff_dsdA(A0,1e-8))
    
    print("Complex Step - dmdA:",complexStep_dmdA(A0,1e-30))
    print("Complex Step - dsdA:",complexStep_dsdA(A0,1e-30))