class Node:
    
    def __init__(self, **kwargs):
        self.id, self.name = kwargs['id'], kwargs['name']
        self.pv = kwargs['pv']
        self.dependency_names = kwargs['dependencies']
        self.resource_reqs = kwargs['resource_reqs']
        self.dependencies = kwargs['dependencies']


from random import randint, random
from math import exp, log
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

class Scheduler:
    
    def __init__(self, projects, resource_constraints, T0=40, c=30, u=10, r=2.0):
        """
        Initialize constants, node values, activations, weights
        """
        self.nodes = enumerate_projects(projects)
        self.resource_constraints = resource_constraints
        self.T0 = float(T0)
        self.N = len(projects)
        self.c = c  # constraint violation penalty
        self.u = u  # utilization penalty
        self.r = r  # 1 + discount rate
        self.init_nodes()
        self.init_weights()
        self.energy_trail = []
        
    def enumerate_projects(projects):
        N = sum([p.duration for p in project])
        projects_dict = {p.name : p for p in projects}
        project_names = sorted(project_dict.keys())
        nodes = []
        i = 0
        name2id = {}; id2node = [0 for _ in range(N)]
    
    for name in project_names:
        p = project_dict[name]
        node = Node(p, dependencies=np.zeros(N))
        for j in range(p.duration):
            id2proj.append(node)
            if j > 0: node.dependencies[j-1] = 1
        name2id[name] = (i, i+p.duration - 1)
        i += p.duration
        
    for i, p in enumerate(id2proj):
        for name in p.dependency_names:
            p.dependencies[name2id[name][1]] = 1
        
    def temp(self, k):
        """
        Return decayed temperature at epoch k
        """
        return self.T0/log(1 + k)
        
    def init_nodes(self):
        """
        Randomly initialize activations, initialize time-decayed values
        """
        self.a = np.random.randint(low=0, high=2, size=(self.N, self.N,))
        self.v = []
        for p in self.projects:
            self.v.append([self.c*sum(p.dependencies) - p.pv*pow(self.r, -j) for j in range(self.N)])
    
    """
    def concurrent(self, i, j):
        "Return True iff i, j are concurrent"
        return bool(self.projects[i].concurrencies[j])
    """
    
    def concurrent(self, i, j):
        return all(sum([np.array(self.projects[k].resource_reqs) for k in [i, j]]) \
            <= np.array(self.resource_constraints))
    
    def dependent(self, i, j):
        """
        Return True iff project i depends on j
        """
        return bool(self.projects[i].dependencies[j])
    
    def weight(self, i, j, m, n):
        """
        Return weight from node (m,n) to (i,j)
        """
        
        if i==m and j==n: return 0
        """
        if j == n:
            
            return self.c * (1 - self.concurrent(i, m)) - self.c * self.dependent(i, m)
        if i == m:
            return self.c"""
        if j==n and self.concurrent(i, m): return 0
        if i==m or j==n: return self.c
        if j > n:
            return -1 * (self.c * self.dependent(i, m))
        return 0
    
    def init_weights(self):
        """
        Initialize weights
        """
        self.weights = []
        for i in range(self.N):
            self.weights.append([])
            for j in range(self.N):
                self.weights[i].append([])
                for m in range(self.N):
                    self.weights[i][j].append([])
                    for n in range(self.N):
                        wt = self.weight(i, j, m, n)
                        #print 'weight from ', i, j, ' to ', m, n, ' is: ', wt
                        self.weights[i][j][m].append(wt)
        #pprint(self.weights)
        self.weights = np.array(self.weights)
        
    def energy(self):
        """
        Return current energy
        """
        e = self.u*max(0.0, float(self.N - self.a.sum()))
        
        for i in range(self.N):
            used = sum([np.array(self.projects[j].resource_reqs) * self.a[i][j] for j in range(self.N)])
            e += self.u * abs(self.resource_constraints - used).sum()
            
            for j in range(self.N):
                e += self.a[i][j]
                e += self.a[i][j]*(self.v[i][j]+(self.a*self.weights[i][j]).sum())
        return e
    
    def interaction_energy(self):
        """
        Return energy from violated constraints
        """
        e = self.u*max(0.0, float(self.N - self.a.sum()))
        for i in range(self.N):
            for j in range(self.N):
                e += self.a[i][j]*(self.a*self.weights[i][j]).sum()
        return e
        
    
    def flip(self, i, j):
        """
        Flip activation of node (i,j)
        """
        self.a[i][j] ^= 1
    
    def energy_delta(self, i, j):
        """
        Return change in energy if activation of (i, j) is flipped
        """
        old = self.energy()
        self.flip(i, j)
        new = self.energy()
        self.flip(i, j)
        return float(new - old)
    
    def npv(self):
        """
        Return current NPV
        """
        return (self.a * self.v).sum()
        
    def flip_prob(self, i, j, k):
        """
        Return prob that node (i,j) flips
        """
        return 1.0/(1 + exp(self.energy_delta(i, j)/self.temp(k)))
        
    def run_epoch(self, k):
        """
        Sequentially run update algorithm for epoch k
        """
        #print self.energy()
        for i in range(self.N):
            for j in range(self.N):
                p = self.flip_prob(i, j, k)
                #print 'prob:', p
                if random() <= p:
                    #print 'flipped to lower energy by ', -1*self.energy_delta(i, j)
                    self.flip(i, j)
                    
    def run_multiple(self, epochs):
        """
        Run multiple epochs
        """
        print 'starting energy: ', self.energy()
        for i in range(epochs):
            self.run_epoch(1 + i)
            self.energy_trail.append(self.energy())
        print 'ending energy: ', self.energy()
            
    def plot_energy_path(self):
        """
        Plot energies through all past epochs
        """
        plt.plot(self.energy_trail)
        plt.show()
        
    def show_soln(self):
        