#Filename:      eightQuGeneticImp.py
#Author:        P. Rivet, 13cpr@queensu.ca
#Date:          21/05/21
#Description:   Solving the infamous 8 queens problem via genetic algo

#Inspired from:
#https://gist.github.com/kushalvyas/7f777c24880c8d1dee744ecb8125e50f
#https://kushalvyas.github.io/gen_8Q.html --> clashes counted wrong?

import numpy as np
import math, random, sys

nQ=8
SHTOP=28
MU=0.8
MU_FLAG=True
MAX_ITER=10000
POPULATION=None

class BoardPos:#object to store information about a particular board layout
    def __init__(self):
        self.seq=None
        self.fit=None
        self.survive=None
    def setSeq(self, val):
        self.seq=val
    def setFit(self, fitness):
        self.fit=fitness
    def setSurvival(self, survival):
        self.survive=survival
    def getAttr(self):
        return {'seq: ':self.seq, 'fitness: ':self.fit, 'survival: ':self.survive}
    
def fitness(chromo=None):
    #conflict testing includes:
        #1. rows
        #2. columns
        #3. diagonals
    #max fitness = 1+2+3+4+5+6+7=28
    #ergo, fval=28-nclash
    nClash=0
    
    #calculate row/col clashes by simply subtracting the array length of
    #unique (non-repeating) positions from the entire array length
    rowColClash=abs(len(chromo)-len(np.unique(chromo)))
    nClash+=rowColClash
    
    #calculate diagonal clashes by iterating through each chromo w/ 2 loops
    #and comparing the index difference with the position difference
    for i in range(len(chromo)):
        for j in range(len(chromo)):
            if i!=j:
                dx=abs(i-j)
                dy=abs(chromo[i]-chromo[j])
                if dx==dy:
                    nClash+=1
                    
    return SHTOP-nClash

def genChromo():#randomly generate a genotype w/ encoded board states
    #global nQ
    #initG=np.arrange(nQ)
    #np.random.shuffle(initG)
    #return initG
    
    #more efficient version of above that doesn't include zeros:
    g=[1,2,3,4,5,6,7,8]
    return np.random.permutation(g)

def genPop(pop_size=100):
    global POPULATION #necessary keyword to modify this variable inside da func
    POPULATION=pop_size
    
    popu=[BoardPos() for k in range(pop_size)]
                
    for l in range(pop_size):
        popu[l].setSeq(genChromo())
        popu[l].setFit(fitness(popu[l].seq))
    
    return popu

def mutate(geno):
    #straightforward elem swap if probability within range
    if (np.random.random()<=MU):
        #generate two random indices for swapping
        #if they're the same, try again
        i1=0
        i2=0
        while(True):
            i1=np.random.randint(1,8)
            i2=np.random.randint(1,8)
            if (i1!=i2):
                break
        temp=geno.seq[i1]
        geno.seq[i1]=geno.seq[i2]
        geno.seq[i2]=temp
                
def xover(par1, par2):#standard single-point crossover, prob=1.0 (modify?)
    chi1=BoardPos()
    chi2=BoardPos()
    xpoint=random.randint(1,7)
    
    #populate genos of children to start
    chi1.seq=genChromo()
    chi2.seq=genChromo()
    
    #1st, copy the segments of the parents before the xpoint to the children
    for m in range(0,xpoint):
        chi1.seq[m]=par1.seq[m]
        chi2.seq[m]=par2.seq[m]
    #2nd, swap the tail of each parent into the other child (1 to 2, 2 to 1)
    for n in range(xpoint, len(par1.seq)):
        chi2.seq[n]=par1.seq[n]
        chi1.seq[n]=par2.seq[n]
    
    return chi1, chi2
        
        
def parentSelect(popu): #run on list of 100 elems BEFORE mutation + xover
    #best 2/5 picked at random survive, ranked by normalized fitness
    globals()
    parents=[]
    sur1, sur2=None, None
    
    #normalization is necessary for ranking purposes
    sumFit=np.sum([g.fit for g in popu])
    for indiv in popu:
        indiv.survive=indiv.fit/(sumFit*1.0)
        
    for a in range(0, 5):
        parents.append(np.random.choice(popu))
        
    #get max fitness, 2nd max
    max1=0
    ndx1=0
    max2=0
    ndx2=0
    for b in range(0,len(parents)):
        if parents[b].survive>max1:
            max1=parents[b].survive
            ndx1=b
    sur1=parents[ndx1]
    parents.pop(ndx1)
    for c in range(0,len(parents)):
        if parents[c].survive>max2:
            max2=parents[c].survive
            ndx1=c
    sur2=parents[ndx2]
    parents.pop(ndx2)
    
    return sur1, sur2
    
def childSelect(popu): #run on list of 100 elems AFTER mutation + xover
    #replace worst 2 after mutation, recombination
    chi1, chi2=None, None
    
    # fitList=[]
    
    # for d in range(0, len(popu)):
    #     fitList.append(popu[d].fit)
        
    # #convert to Numpy array for optimized numeric sorting
    # fitArr=np.array(fitList)
    # np.ndarray.sort(fitArr)#NON-UNIQUE??
    
    #above approach is more efficient but also more annoying to implement
    
    #get min fitness, 2nd min
    min1=1
    ndx1=0
    min2=1
    ndx2=0
    for b in range(0,len(popu)):
        try:
            if popu[b].survive<min1:
                min1=popu[b].survive
                ndx1=b
        except TypeError:
            continue
    popu.pop(ndx1)#remove min after loop finishes
    for c in range(0,len(popu)):
        try:
            if popu[c].survive<min2:
                min2=popu[c].survive
                ndx2=c
        except TypeError:
            continue
    popu.pop(ndx2)#remove 2nd min
    
    return popu #returned list now has 98 instead of 100 elems
                #after this function runs, the next iter of the GA begins

def GA(popu):
    n_iter=1
    #main loop which runs until either of the stop conditions are met
    while(True):
        
        print('EXECUTING 8Q GENETIC ALGORITHM. ITERATION NUMBER: '+str(n_iter))
        
        #step 1: parent selection
        for l in range(0, len(popu)):#re-compute fitness to avoid bugs
            popu[l].setFit(fitness(popu[l].seq))
        par1, par2=parentSelect(popu)
        
        #step 2: variation
        mutate(par1) #mutation arbitrarily set before xover but the inverse
        mutate(par2) #is also possible - faster convergence?
        
        chi1, chi2=xover(par1, par2)
        
        popu.append(chi1)
        popu.append(chi2)
        
        #step 3: child selection
        popu=childSelect(popu)
        
        #evaluate fitness at end of current iter
        fitnessvals=[pos.fit for pos in popu]
        if SHTOP in fitnessvals:
            break
        elif n_iter==MAX_ITER:
            break
        else:
            n_iter+=1
            continue
        
    return popu

p1=BoardPos()
p2=BoardPos()  
p1.seq=genChromo()
p2.seq=genChromo()
chi1, chi2=xover(p1, p2)

popu=genPop()
GA(popu)

print('Correct solution value(s) achieved: ')
for soln in popu:
    if soln.fit==28:
        print(soln.seq)
print('Maximum fitness values achieved: ')
fitnessvals=[pos.fit for pos in popu]
print(fitnessvals)
