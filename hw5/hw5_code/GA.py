import numpy as np

class GA():
    def __init__(self,lower_bounds,upper_bounds,num_dim=5,pop_size=None,max_itr=500,mut_std_dev=0.001):
        self.lb = lower_bounds
        self.ub = upper_bounds
        if not pop_size:
            self.pop_size = num_dim * 10
        else:
            self.pop_size = pop_size
        self.ndim = num_dim
        self.max_itr = max_itr
        self.mutation_std_dev = mut_std_dev
        print("lb:",self.lb,"ub:",self.ub,\
            "pop_size:",self.pop_size,"ndim:",self.ndim,\
            "max_itr:",self.max_itr,"std_dev:",self.mutation_std_dev)

    def run(self,obj,args):
        self.args = args
        Pk = self.generateInitPop()
        for k in range(self.max_itr):
            f_list = self.computeF(obj,Pk)
            parents, parents_score = self.selection(f_list,Pk)
            offspring = self.crossover(parents, parents_score)
            Pk = self.mutate(offspring)
        best = self.pickBest(obj,Pk)
        return best
    
    def generateInitPop(self):
        # Random population initialization
        Pk =  np.random.uniform(self.lb,self.ub,size=(self.pop_size,self.ndim))
        Pk_idxs = np.argsort(np.linalg.norm(Pk, axis=1))
        return Pk[Pk_idxs,:]

    def computeF(self,obj,Pk):
        f_list = []
        for i in range(np.size(Pk,0)):
            f_list.append(obj(Pk[i,:],self.args))
        return f_list

    def selection(self,f_list,Pk):
        # Tournament Approach
        parents = []
        parents_score = []
        for m in range(2):
            idx1 = np.random.choice(np.arange(self.pop_size),size=int(self.pop_size/2))
            idx2 = np.random.choice(np.arange(self.pop_size),size=int(self.pop_size/2))
            for i in range(int(self.pop_size/2)):
                if f_list[idx1[i]] < f_list[idx2[i]]:
                    # If 1st index is more fit
                    parents.append(Pk[idx1[i],:])
                    parents_score.append(f_list[idx1[i]])
                else:
                    parents.append(Pk[idx2[i],:])
                    parents_score.append(f_list[idx2[i]])
        return np.array(parents), parents_score
    
    def crossover(self,parents,parents_score):
        offspring = []
        idx1 = np.random.choice(np.arange(self.pop_size),size=int(self.pop_size/2))
        idx2 = np.random.choice(np.arange(self.pop_size),size=int(self.pop_size/2))
        for i in range(int(self.pop_size/2)):
            child1 = 0.5 * parents[idx1[i],:] + 0.5 * parents[idx2[i],:]
            offspring.append(self.checkBounds(child1))
            if parents_score[idx1[i]] < parents_score[idx2[i]]:
                child2 = 2.0 * parents[idx1[i],:] - parents[idx2[i],:]
            else:
                child2 = - parents[idx1[i],:] + 2.0 * parents[idx2[i],:]
            offspring.append(self.checkBounds(child2))
        return np.array(offspring)
    
    def checkBounds(self,x):
        for i in range(self.ndim):
            if x[i] < self.lb:
                x[i] = self.lb
            elif x[i] > self.ub:
                x[i] = self.ub
        return x

    def mutate(self,offspring):
        offspring += np.random.uniform(0,self.mutation_std_dev,size=(self.pop_size,self.ndim))
        for i in range(np.size(offspring,0)):
            offspring[i,:] = self.checkBounds(offspring[i,:])
        return offspring

    def pickBest(self,obj,Pk):
        f_list = self.computeF(obj,Pk)
        best = np.inf
        best_idx = -1
        for i in range(np.size(Pk,0)):
            if f_list[i] < best:
                best = f_list[i]
                best_idx = i
        return Pk[best_idx,:]