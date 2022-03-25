import numpy as np
from scipy.optimize import minimize
from wavedrag import wavedrag

'''
#data:
f = np.array([7.7859, 5.9142, 5.3145, 5.4135, 1.9367, 2.1692, 0.9295, 1.8957, -0.4215, 0.8553, 1.7963, 3.0314, 4.4279, 4.1884, 4.0957, 6.5956, 8.2930, 13.9876, 13.5700, 17.7481])
x = np.array([-2.0000, -1.7895, -1.5789, -1.3684, -1.1579, -0.9474, -0.7368, -0.5263, -0.3158, -0.1053, 0.1053, 0.3158, 0.5263, 0.7368, 0.9474, 1.1579, 1.3684, 1.5789, 1.7895, 2.0000])

weights = np.ones(3)
powers = np.array([0, 1, 2])
# powers = np.arrange(0, np.shape(x)[0])


ones = np.ones_like(x)
psi = np.array([x**2, x, ones]).T
w = np.linalg.inv(psi.T@psi)@psi.T@f
print(w)

plt.figure(1)
plt.scatter(x, f, c='b')

x_vals = np.arange(start=-2, stop=2, step=0.1)
y_vals = w[0]*x_vals**2 + w[1]*x_vals+w[2]
plt.plot(x_vals, y_vals, c='g')

plt.show()
'''

def getPsi(x):
    # 6 dim, quadratic model
    Psi = np.array([1])
    Psi = np.append(Psi,x[:])
    Psi = np.append(Psi,x[0]*x[:])
    Psi = np.append(Psi,x[1]*x[1:])
    Psi = np.append(Psi,x[2]*x[2:])
    Psi = np.append(Psi,x[3]*x[3:])
    Psi = np.append(Psi,x[4]*x[4:])
    Psi = np.append(Psi,x[5]*x[5])
    Psi = np.expand_dims(Psi,axis=1)
    # print(Psi)
    # print(Psi.shape)
    return Psi

def getSurrogate(xi,fi):
    # fi_hat = []
    wi = []
    for i in range(xi.shape[0]):
        Psi = getPsi(xi[i,:])
        w = np.linalg.inv(Psi.T @ Psi) @ Psi.T * fi[i]
        wi.append(w)
        # fi_hat.append((w @ Psi)[0,0])
    return w

def fhat(x,*args):
    w = args
    return (w @ getPsi(x))[0,0]


if __name__ == '__main__':
    kmax = 100
    tau = 1e-1
    
    xl = np.array([0.05,0.15,0.65,1.75,3,4.25])
    xu = np.array([0.25,0.4,0.85,2.25,3.25,4.75])

    # Sample 20 points
    xi = np.random.uniform(low=xl, high=xu, size=(20,6))
    fi = []
    for i in range(xi.shape[0]):
        fi.append(wavedrag(xi[i,:]))
    print(fi)

    k = 0
    while k < kmax:
        wi = getSurrogate(xi,fi)
        x = minimize(fhat,xi[0,:],args=wi)
        x_star = x.x
        fnew = x.fun
        