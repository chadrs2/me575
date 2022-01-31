from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

def plot_f(ans):
    x1 = np.arange(-2,2,0.01)
    x2 = np.arange(-1,2,0.01)
    X1, X2 = np.meshgrid(x1,x2)
    Z1 = (1-X1)**2+100*(X2-X1**2)**2

    fig, axs = plt.subplots()
    levels = np.arange(0,50,1)
    CS = axs.contour(X1,X2,Z1,levels=levels)
    CB = fig.colorbar(CS)
    axs.scatter(ans[0],ans[1],c='r')
    axs.set_xlabel("X1")
    axs.set_ylabel("X2")
    plt.show()

def constraint1(x):
    return 1-(x[0]**2+x[1]**2)

def constraint2(x):
    return 5-(x[0]+3*x[1])

if __name__ == '__main__':
    x0 = np.array([0,0])
    print("Initial Guess:",x0)

    cons = [
        {'type':'ineq','fun':constraint1},
        {'type':'ineq','fun':constraint2}
    ]

    x = minimize(f,x0,constraints=cons)
    print("Solution:",x)

    plot_f(x.x) 