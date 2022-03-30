import numpy as np
from scipy.optimize import minimize, Bounds
from wavedrag import wavedrag


def getPsi(x):
    # 6 dim, quadratic model
    quadratic_cols = int(1+x.shape[1]+x.shape[1]*(x.shape[1]+1)/2)
    Psi = np.zeros((x.shape[0],quadratic_cols))
    for i in range(x.shape[0]):
        Psi_new = np.array([1])
        Psi_new = np.append(Psi_new,x[i,:])
        Psi_new = np.append(Psi_new,x[i,0]*x[i,:])
        Psi_new = np.append(Psi_new,x[i,1]*x[i,1:])
        Psi_new = np.append(Psi_new,x[i,2]*x[i,2:])
        Psi_new = np.append(Psi_new,x[i,3]*x[i,3:])
        Psi_new = np.append(Psi_new,x[i,4]*x[i,4:])
        Psi_new = np.append(Psi_new,x[i,5]*x[i,5])
        Psi[i,:] = Psi_new
    return Psi

def getSurrogate(xi,fi):
    Psi = getPsi(xi)
    w = np.linalg.inv(Psi.T @ Psi) @ Psi.T @ fi
    return w

def fhat(x,*args):
    w = args[0]
    return (w @ getPsi(np.expand_dims(x,axis=0)).T)[0]


if __name__ == '__main__':
    # Input Params
    xl = np.array([0.05,0.15,0.65,1.75,3,4.25])
    xu = np.array([0.25,0.4,0.85,2.25,3.25,4.75])
    kmax = 20
    tau = 5e-2
    ns = 20

    num_func_calls = 0

    # Sample 20 points
    xi = np.random.uniform(low=xl, high=xu, size=(ns,6))
    fi = []
    for i in range(xi.shape[0]):
        fi.append(wavedrag(xi[i,:]))
        num_func_calls += 1

    f_new = fi[0]
    fhat_star = f_new - 1
    k = 0
    # print("K:",k,"err:",((f_new - fhat_star) / f_new))
    # while k < kmax and np.linalg.norm(f_new-fhat_star) > tau: #((f_new - fhat_star) / f_new) > tau:
    while k < kmax and np.abs(((f_new - fhat_star) / f_new)) > tau:
        # print("K:",k,"Coefficient of Drag Error:",((f_new - fhat_star) / f_new))
        # print("Error:",np.linalg.norm(f_new-fhat_star))
        w = getSurrogate(xi,fi)
        x = minimize(fhat,xi[-1,:],args=w,bounds=Bounds(xl,xu,keep_feasible=True))
        x_star = x.x
        fhat_star = x.fun
        f_new = wavedrag(x_star)
        num_func_calls += 1
        xi = np.append(xi, np.expand_dims(x_star,axis=0), axis=0)
        fi = np.append(fi, f_new)
        k += 1
        # print("After: K:",k,"err:",((f_new - fhat_star) / f_new))
        # print("Post Error:",np.linalg.norm(f_new-fhat_star))
        # print("------------------------------------------------------")

    print("Surrogate-Optimized Minimum Drag:",fhat_star)
    print("Surrogate-Optimized Number of Function Calls:",num_func_calls)


    # Running optimization on actual function
    x = minimize(wavedrag, xi[0,:], bounds=Bounds(xl,xu,keep_feasible=True))
    print("Scipy Optimize Minimum Drag:",x.fun)
    print("Scipy Optimize Number of Function Calls:",x.nfev)


