import numpy as np
import matplotlib.pyplot as plt

from GA import GA

def obj(X,xf,n):
    X = X.reshape((n,2))
    dist_sum = 0
    for i in range(np.size(X,0)):
        dist_sum += np.linalg.norm(X[i,:] - xf)
    return dist_sum

def anchor_con(X,x0,d,n):
    X = X.reshape((n,2))
    constraint = []
    for i in range(np.size(X,0)):
        if i == 0:
            constraint.append(np.linalg.norm(X[i,:]-x0)-d)
        else:
            constraint.append(np.linalg.norm(X[i,:]-X[i-1,:])-d)
    return np.array(constraint)

def obst_con(X,centers,r,n):
    X = X.reshape((n,2))
    constraint = []
    for i in range(np.size(X,0)):
        for c in  centers:
            # constraint.append(r-np.linalg.norm(X[i,:]-c))
            constraint.append(r-np.linalg.norm(X[i,:]-c)+np.random.normal(0,0.1)) # Noise of std_dev=10cm
    return np.array(constraint)

def obj_penalty(X,*args):
    n, x0, xf, d, c, r, mu = args[0][0], args[0][1], args[0][2], args[0][3], args[0][4], args[0][5], args[0][6]

    f = obj(X,xf,n)
    h_sum = np.sum(np.square(anchor_con(X,x0,d,n)))
    gx = obst_con(X,c,r,n)
    g_sum = np.sum(np.square(np.maximum(np.zeros_like(gx),gx)))
    
    f_hat = f + mu[0]/2 * h_sum + mu[1]/2 * g_sum
    return f_hat 


if __name__ == '__main__':
    # Initial Conditions
    # X = np.array([[1,1]])
    X = np.array([
        [1,1],
        [2,2],
        [3,3],
        [4,4],
        [5,5]
    ])
    xf = np.array([10,10])
    x0 = np.array([0,0])
    step_size = 0.5
    centers = [np.array([4,3.5]),
                np.array([7.5,7.5]),
                np.array([6,6.5])]
    r = 1.
    # max_size = 10
    # num_circles = 6
    # max_rad = 2
    # min_rad = 0.5
    # centers = np.around(np.random.uniform(2,max_size-max_rad, (num_circles,2)), 1)
    # r = np.around(np.random.uniform(min_rad, max_rad, num_circles),1)
    
    # GA Init
    print("X0:",X)
    # # print("lb:",np.min(x0),"ub:",1.5*np.max(xf),"pop_size:",100,"ndim:",np.size(X),"max_itr:",100,"std_dev:",0.1)
    # ga = GA(np.min(x0),1.5*np.max(xf),np.size(X),
    #         pop_size=100,
    #         max_itr=100,
    #         mut_std_dev=0.1)
    # print("lb:",np.min(x0),"ub:",1.5*np.max(xf),"pop_size:",10*np.size(X),"ndim:",np.size(X),"max_itr:",100,"std_dev:",0.1)
    ga = GA(np.min(x0),1.5*np.max(xf),np.size(X),
            pop_size=10*np.size(X),
            max_itr=100,
            mut_std_dev=0.1)    

    # Quadratic Penalty Method Variables
    n = np.size(X,0)
    m = 2 # 1 for equality and 1 for inequality constraints
    
    path = np.array([x0[:]])
    tau = 2*step_size
    is1stIteration = True
    itr = 0
    while np.linalg.norm(X[0,:] - xf) > tau:
        if not is1stIteration:
            X = np.vstack([X[1:,:],X[-1,:]+step_size])
        mu = np.ones((m,)) * 1 # mu > 0
        rho = np.ones((m,)) * 2 # Ext: rho > 1, Int: rho < 1

        epsilon = 1e8
        # itr = 0
        X = X.reshape(n*2,)
        while np.min(mu) < epsilon:
            args = [n,x0,xf,step_size,centers,r,mu]
            x = ga.run(obj_penalty,args)
            # x = minimize(obj_penalty,X,args=args,method='Nelder-Mead',tol=1e-1)
            mu = rho * mu # mu gets larger each time

        X = x.reshape(n,2)
        path = np.vstack([path,X[0,:]])

        if is1stIteration:
            # true_optimum = step_size * np.array([
            #     [np.sqrt(2)/2,np.sqrt(2)/2],
            #     [2*np.sqrt(2)/2,2*np.sqrt(2)/2],
            #     [3*np.sqrt(2)/2,3*np.sqrt(2)/2]
            # ])
            # print("Accuracy:",np.linalg.norm(X-true_optimum),"meters")
            # print("Number of iterations until convergence:",itr)
            is1stIteration = False

        if itr % 5 == 0:
            fig, axs = plt.subplots()
            for i in range(len(centers)):
                if i == 0:
                    circle_i = plt.Circle(centers[i],radius=r,color='r',fill=True,label="Obstacle(s)")
                else:
                    circle_i = plt.Circle(centers[i],radius=r,color='r',fill=True)
                axs.add_patch(circle_i)
            axs.plot(path[:,0],path[:,1],'g+',label="Path")
            axs.scatter(X[:,0],X[:,1],c='g',marker='.',label="Horizon")
            axs.plot(0,0,'r*',label="Starting Position")
            axs.plot(xf[0],xf[1],'b*',label="Ending Position")
            axs.set_title("Receding Horizon Path Planning GA and Penalty Methods")
            axs.set_xlabel("X (m)")
            axs.set_ylabel("Y (m)")
            axs.legend()
            plt.show()

        # Update variables (i.e. move along path)
        x0 = X[0,:]
        print("Current Position:",x0,"itr:",itr)
        itr += 1

    fig, axs = plt.subplots()
    # for i in range(len(centers)):
    #     if i == 0:
    #         circle_i = plt.Circle(centers[i],radius=r,color='r',fill=True,label="Obstacle(s)")
    #     else:
    #         circle_i = plt.Circle(centers[i],radius=r,color='r',fill=True)
    #     axs.add_patch(circle_i)
    for i in range(len(centers)):
        if i == 0:
            circle_i = plt.Circle(centers[i],radius=r,color='r',fill=False,label="Boundary")
        else:
            circle_i = plt.Circle(centers[i],radius=r,color='r',fill=False)
        axs.add_patch(circle_i)
    
    for i in range(len(centers)):
        r1 = (1-0.341)*r
        if i == 0:
            circle_i = plt.Circle(centers[i],radius=r1,color='b',fill=True,label="True Obstacle(s)")
        else:
            circle_i = plt.Circle(centers[i],radius=r1,color='b',fill=True)
        axs.add_patch(circle_i)
    axs.plot(path[:,0],path[:,1],'g+',label="Path")
    axs.plot(0,0,'r*',label="Starting Position")
    axs.plot(xf[0],xf[1],'b*',label="Ending Position")
    axs.set_title("Receding Horizon Path Planning Using GA and Penalty Methods")
    axs.set_xlabel("X (m)")
    axs.set_ylabel("Y (m)")
    axs.legend()
    plt.show()