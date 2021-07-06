#Filename:      TSPGeneticImp.py
#Author:        P. Rivet, 13cpr@queensu.ca
#Date:          21/06/22
#Description:   Solving the infamous travelling salesman problem via genetic algo
#               Distances to be minimized according to the fitness measure
#               are read from a .csv file in adjacency matrix format

#https://github.com/satvik-tiwari/Genetic-Algorithm/blob/master/Travelling%20Salesman%20Problem/TSP.py

#https://jaketae.github.io/study/genetic-algorithm/ 

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format='svg'
#above 2 lines only work in a Jupyter notebook
plt.style.use("seaborn")
np.random.seed(42) #added for reproducibility

data=pd.read_csv('cities.csv')
distances=data.iloc[:,1:].values
sz=distances.shape
data.head()
distances=np.asarray(distances)
cities=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

class Popu(): #stuff all methods into this one object for convenience
    def __init__(self, bog, adj_mat):
        self.bog=bog #full population which is a subset of the adjacency matrix
        self.parents=[]
        self.score=0 #fitness function
        self.best=None
        self.adj_mat=adj_mat
        
    def fitness(self, chromo): #sum of all distances between 2 adjacent cities
        return sum([self.adj_mat[chromo[i], chromo[i+1]] for i in range(len(chromo)-1)])
    
    def evaluate(self):
        dist=np.asarray([self.fitness(chromo) for chromo in self.bog]) #fitness vec
        self.score=np.min(dist) #lower value is better (shorter distance)
        self.best=self.bog[dist.tolist().index(self.score)] #location in array
        try:
            self.parents.append(self.best) #add to parent
        except AttributeError:
            for j in range(len(self.best)):
                np.append(self.parents, self.best[j])
        
        if False in (dist[0]==dist): #prevent worst chromo from being selected
            dist=np.max(dist)-dist
        
        return dist/np.sum(dist) #probability vector representing the chance that
                                 #each elem is chosen as a parent based on its calculated fitness
                                 
    def select(self, k=8): #roulette model -> prob. vector compared to uniform sample
        fit=self.evaluate()
        
        while len(self.parents)<k:
            idx=np.random.randint(0,len(fit))
            if fit[idx]>np.random.rand():
                self.parents.append(self.bog[idx])
        
        self.parents=np.asarray(self.parents)
        
        #2-point crossover for greater variability
    def xover(self, pc=0.1): 
        childoos=[]
        cnt, sz=self.parents.shape
        
        for _ in range(len(self.bog)):
            
            #if random value is greater than crossover prob. then simply 
            #append an arbitrarily chosen parent to the child list
            if np.random.rand()>pc:
                childoos.append(list(self.parents[np.random.randint(cnt, size=1)[0]]))
            #otherwise, generate 2 children via two-point xover
            else:
                par1, par2=self.parents[np.random.randint(cnt, size=2), :]
                idy=np.random.choice(range(sz), size=2, rep=False)
                start, end=min(idy), max(idy) #randomly generated slice of geno to cross
                child=[None]*sz
                
                for j in range(start, end+1, 1):
                    child[j]=par1[j]
                pnt=0 #pointer to keep track of progress along the data structure
                
                for k in range(sz):
                    if child[k] is None:
                        while par2[pnt] in child:
                            pnt+=1
                        child[k]=par2[pnt]
                childoos.append(child)
                
            return childoos
        
        #eazy breezy element swap
    def mutate (chromo):
        x, y = np.random.choice(len(chromo), 2)
        chromo[x], chromo[y]=chromo[y], chromo[x]
        
        return chromo
        
    #wrapper function is more convenient calling-wise
    def variation (self, pc=0.1, pm=0.1):
        nextBog=[]
        chili=self.xover(pc)
        
        for chi in chili:
            if np.random.rand() < pm:
                nextBog.append(self.mutate(chi))
            else:
                nextBog.append(chi)
                
        return nextBog
            
    
def init_pop(cities, adj_mat, n_pop): #randomly permute the specified number of genotypes
    return Popu(np.asarray([np.random.permutation(cities) for _ in range(n_pop)]), adj_mat)

def GA(cities, adj_mat, n_pop=10, n_iter=1000, selective=0.2, pc=0.5, pm=0.05, print_interv=100, returnHist=False):
    popu=init_pop(cities, adj_mat, n_pop)
    best=popu.best
    skor=float("inf")
    hist=[]
    
    for i in range(n_iter):
        popu.select(n_pop*selective)
        hist.append(pop.score)
        
    print(f"Gen {i}: {popu.score}")
    
    if popu.score<skor:
        best=popu.best
        score=popu.score
        
    kids=popu.variation(pc, pm)
    popu=Popu(kids, popu.adj_mat)
    
    if returnHist:
        return best, hist
    
    return best
        
    

pop=init_pop(cities, distances, 10)
#pop.fitness=fitness #assigning a new function in this way seems sloppy...
print(pop.evaluate())

pop.select()
print(pop.parents)

print(pop.variation())


GA(cities, distances, True)