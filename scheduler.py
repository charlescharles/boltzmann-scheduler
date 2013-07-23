from random import randint, random
from math import exp, log
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

class Project:
    
    def __init__(self, **kwargs):
        self.id, self.name = kwargs['id'], kwargs['name']
        self.pv = kwargs['pv']
        self.dependencies = kwargs['dependencies']
        self.concurrencies = kwargs['concurrencies']


class Scheduler:
    
    def __init__(self, projects, T0=30, c=30, u=10, r=1.5):
        """
        Initialize constants, node values, activations, weights
        """
        self.projects = projects
        self.T0 = float(T0)
        self.N = len(projects)
        self.c = c
        self.u = u
        self.r = r
        self.init_nodes()
        self.init_weights()
        self.energy_trail = []
        
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
            
    def concurrent(self, i, j):
        """
        Return True iff i, j are concurrent
        """
        return bool(self.projects[i].concurrencies[j])
    
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
        if j == n:
            return self.c * (1 - self.concurrent(i, m)) - self.c * self.dependent(i, m)
        if i == m:
            return self.c
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
            for j in range(self.N):
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