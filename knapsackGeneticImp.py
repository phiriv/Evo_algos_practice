#Filename:      knapsackGeneticImp.py
#Author:        P. Rivet, 13cpr@queensu.ca
#Date:          21/06/15
#Description:   Solving the infamous knapsack problem via genetic algo
#               inspired by https://gist.github.com/satvik-tiwari/068cc0348b76945007795014daa671ad#file-ga_2-1-py

import math, random, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nK=500
STOP=25
MU=0.002
XOVER=0.7
MAX_ITER=10000
POPULATION=None

#perform crossover for specified number of offpsring
def xover (par, nOffspring):
    offspring=np.empty([nOffspring, par.shape[1]])
    xPoint=random.randint(1,nK)#modified from always being halfway
    
    k=0
    while(par.shape[0]<nOffspring):
        x=np.random.random()
        if x<=XOVER:
            p1ind=k%par.shape[0]
            p2ind=(k+1)%par.shape[0]
            offspring[k,0:xPoint]=par[p1ind,0:xPoint]
            offspring[k,xPoint:]=par[p2ind,xPoint:]
            k+=1
            
    return offspring
    
#MUTATIS MUTANDIS
def mutate(pop):
    mu=np.empty(pop.shape)
    
    for i in range(mu.shape[0]):
        x=np.random.random()
        mu[i,:]=pop[i,:]
        
        if x<=MU:
            #generate random value for bit flip
            rnd=np.random.randint(0,pop.shape[0]-1)
            if mu[i, rnd]==0:
                mu[i, rnd]==1
            else:
                mu[i, rnd]==0
    
    return mu
    
#calculate fitness based on input weights, values, candidates, constraint
def getFitness(w, v, pop, c):
    fit=np.empty(pop.shape[0])
    
    for j in range(pop.shape[0]):
        s1=np.sum(pop[j]*v)
        s2=np.sum(pop[j]*w)
        if s2<c:
            fit[j]=s1
        else:
            fit[j]=0
            
    return fit

#execute selection with fitness values, number of parents, population
def select(fit, nPar, pop):
    
    fit=list(fit)
    par1=np.empty([pop.shape[0],nPar])
    
    for l in range(pop.shape[0]):
        maxFitNdx=np.where(fit==np.max(fit))
        par1[l,:]=pop[maxFitNdx[0][0],:]
        fit[maxFitNdx[0][0]]=-1
    
    return par1

#run genetic algorithm w/ specified params
def GA(w, v, pop, pSz, nGen, c):
    params, fitH=[],[]
    
    for m in range(nGen):
        fit=getFitness(w, v, pop, c)#not single value?
        fitH.append(fit)
        par=select(fit, pSz, pop)
        offspring=xover(par, 2)
        mutants=mutate(offspring)
        pop[0:par.shape[0],:]=par#WRONG DIMS!?
        pop[:mutants.shape[0],:]=mutants
    
    lastFit=getFitness(w, v, pop, c)
    maxFit=np.where(lastFit==np.max(lastFit))
    params.append(pop[maxFit[0][0],:])
    
    return params, fitH

#randomly initialize starting population
items=np.arange(1,nK+1)
weights=np.random.randint(1,20, size=nK)
values=np.random.randint(1,1000, size=nK)
cmax=100 #max carrying capacity

nSolns=50
popSz=(nSolns, items.shape[0])
print('POPULATION SIZE: {}'.format(popSz))

initPop=np.random.randint(2,size=popSz)
initPop=initPop.astype(int)
print('Initial population: \n{}'.format(initPop))

#RUN
#add later: wrapper for n times
params, fitHist=GA(weights, values, initPop, nK, MAX_ITER, cmax)
print('Optimal parameters after running GA: \n{}'.format(params))

goodItems=items*params
print('\nHighest value items within the constraint of {}:'.format(cmax))
for n in range(items.shape[0]):
    if items[0][n]!=0:
        print('{}\n'.format(items[0][n]))