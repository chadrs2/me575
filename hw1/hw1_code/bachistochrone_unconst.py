from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

def f(y, *args):
    n = args[0]
    x = args[1]
    mu_k = args[2]
    h = args[3]
    tot_travel_time = 0
    # Fix endpoints
    y[0] = 1
    y[-1] = 0
    # Compute total travel time by summing each line along curve
    for i in range(0,n-1):
        num = np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2)
        denom = np.sqrt(h-y[i+1]-mu_k*x[i+1]) + np.sqrt(h-y[i]-mu_k*x[i])
        tot_travel_time += num / denom
    return tot_travel_time

if __name__ == '__main__':
    h = 1 # starting y-position
    mu_k = 0.3 # coefficient of kinetic friction
    n = 12  # start with n points
    g = 9.81 # gravity (m/s^2)

    Y0 = np.linspace(h,0,num=n,endpoint=True)
    X0 = np.linspace(0,1,num=n,endpoint=True)
    
    args = (int(n), X0, mu_k, h)
    Y = minimize(f,Y0,args=args)#,constraints=cons)
    print(Y)

    # (a) Plot the optimal shape with n = 12 (10 design variables).
    plt.scatter(X0,Y.x)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bachistochrone Solution for n ="+str(n))
    # plt.show()
    
    # (b) Report the travel time between the two points. Don’t forget to put g = 9.81, the acceleration of
    # gravity, back in.
    print("Total Travel time:",np.sqrt(2/g)*Y.fun,"sec")
    plt.show()

    # (c) Study the effect of increased problem dimensionality. Start with 4 points and double the dimension
    # up to 128 (i.e., 4, 8, 16, 32, 64, 128). Plot and discuss the increase in computational expense
    # with problem size. Example metrics include things like major iterations, functional calls, wall
    # time, etc. Hint: When solving the higher-dimensional cases, it is more effective to start with the
    # solution interpolated from a lower-dimensional case—this is called a warm start.
    num_itr = []
    n_list = [4,8,16,32,64,128]
    Y0 = np.linspace(h,0,num=4,endpoint=True)
    X0 = np.linspace(0,1,num=4,endpoint=True)
    for n in n_list:
        args = (int(n), X0, mu_k, h)
        Y = minimize(f,Y0,args=args)
        print(Y)
        num_itr.append(Y.nit)

        # Plot solution
        plt.scatter(X0,Y.x)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Bachistochrone Solution for n ="+str(n))
        plt.show()

        # Compute & Print Total Travel Time from point 0 to point n
        print("Total Travel time:",np.sqrt(2/g)*Y.fun,"sec")

        # Interpolating for Re-initialization
        if n != 128:
            f_of_x = interp1d(X0,Y.x)
            X0 = np.linspace(0,1,num=2*n,endpoint=True)
            Y0 = f_of_x(X0)
    
    # Plot effects of dimensionality based on varying values of n
    plt.figure()
    plt.scatter(n_list,num_itr)
    plt.title("Dimensionality Effects")
    plt.xlabel("n")
    plt.ylabel("# Iterations")
    plt.show()