import numpy as np


def obj(x,beta):
    return x[0]**2+x[1]**2-beta*x[0]*x[1]

def gradient(x,beta):
    return np.array([2*x[0]-beta*x[1],2*x[1]-beta*x[0]])

if __name__ == '__main__':
    beta = 3./2.
    x = np.array([1,1])

    prev_alpha = 1e-1
    prev_delta_f = gradient(x,beta)
    prev_p = - prev_delta_f
    first_itr = True
    tau = 1e-6
    # Steepest Descent
    while np.linalg.norm(gradient(x,beta),np.inf) > tau:
        if first_itr:
            x = x + prev_alpha * prev_p
            first_itr = False
        else:
            delta_f = gradient(x,beta)
            p = -delta_f / np.linalg.norm(delta_f)
            alpha = prev_alpha * (prev_delta_f.T @ prev_p) / (delta_f.T @ p)
            x = x + alpha * p

            prev_alpha = alpha
            prev_p = p
            prev_delta_f = delta_f
            print(x)

    #x*=(0,0)
    print("x*:",x)