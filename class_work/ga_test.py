import numpy as np
from geneticalgorithm import geneticalgorithm as ga

def rosenbrock(x):
    Sigma = 0
    for i in range(len(x)-1):
        Sigma += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return Sigma


def f(X):
    return np.sum(X)

ndim = 10
lb = -1
ub = 3
varbound=np.array([[lb,ub]]*ndim)

# variable_type = 'int' for discrete and 'real' for continuous
# algorithm_param = {'max_num_iteration': 3000,\
#                    'population_size':100,\
#                    'mutation_probability':0.1,\
#                    'elit_ratio': 0.01,\
#                    'crossover_probability': 0.5,\
#                    'parents_portion': 0.3,\
#                    'crossover_type':'uniform',\
#                    'max_iteration_without_improv':None}
# model=ga(function=f,\
#             dimension=3,\
#             variable_type='real',\
#             variable_boundaries=varbound,\
#             algorithm_parameters=algorithm_param)

model=ga(function=rosenbrock,dimension=ndim,variable_type='real',variable_boundaries=varbound)

model.run()