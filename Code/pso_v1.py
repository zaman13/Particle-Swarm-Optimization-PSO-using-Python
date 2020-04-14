# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:48:18 2020

@author: Mohammad Asif Zaman
Particle swarm optimization code 
- General code, would work with fitness function of any dimensions (any no. of parameters)
- Vectorized fast code. Only one for loop is used to go over the iterations. Calculations
  over all the dimensions and particles are done using matrix operations.
- One function call per iteration.  

Tested in python 2.7.
"""
from __future__ import print_function    

import time
import math
import numpy as np
import pylab as py


# Control parameters
w = 0.5                   # Intertial weight
c1 = 2.0                  # Weight of searching based on the optima found by a particle
c2 = 2.0                  # Weight of searching based on the optima found by the swarm

Np = 40                   # population size (number of particles)
D = 4                     # dimension (= no. of parameters in the fitness function)
max_iter = 100            # maximum number of iterations 
xL = np.zeros(D) - 4      # lower bound (does not need to be homogeneous)  
xU = np.zeros(D) + 4      # upper bound (does not need to be homogeneous)   



# Fitness function. The code maximizes the value of the fitness function
def fitness(x):
    # x is a matrix of size D x Np
    # The position of the entire swarmp is inputted at once. 
    # Thus, one function call evaluates the fitness value of the entire swarm
    # F is a vector of size Np. Each element represents the fitness value of each particle in the swarm
    
    F_sphere = 2.0 - np.sum(np.multiply(x,x),0)    # modified sphere function
    return F_sphere



pbest_val = np.zeros(Np)            # Personal best fintess value. One pbest value per particle.
gbest_val = np.zeros(max_iter)      # Global best fintess value. One gbest value per iteration (stored).


pbest = np.zeros((D,Np))            # pbest solution
gbest = np.zeros(D)                 # gbest solution

gbest_store = np.zeros((D,max_iter))   # storing gbest solution at each iteration

x = np.random.rand(D,Np)            # Initial position of the particles
v = np.zeros((D,Np))                # Initial velocity of the particles


# Setting the initial position of the particles over the given bounds [xL,xU]
for m in range(D):    
    x[m,:] = xL[m] + (xU[m]-xL[m])*x[m,:]
    

# Function call. Evaluates the fitness of the initial swarms    
fit = fitness(x)           # vector of size Np

pbest_val = np.copy(fit)   # initial personal best = initial fitness values. Vector of size Np
pbest = np.copy(x)         # initial pbest solution = initial position. Matrix of size D x Np

# Calculating gbest_val and gbest. Note that gbest is the best solution within pbest                                                                                                                      
ind = np.argmax(pbest_val)                # index where pbest_val is maximum. 
gbest_val[0] = np.copy(pbest_val[ind])    # set initial gbest_val
gbest = np.copy(pbest[:,ind])


print("Iter. =", 0, ". gbest_val = ", gbest_val[0])
print("gbest_val = ",gbest_val[0])


# Loop over the generations
for iter in range(1,max_iter):
    
  
    r1 = np.random.rand(D,Np)           # random numbers [0,1], matrix D x Np
    r2 = np.random.rand(D,Np)           # random numbers [0,1], matrix D x Np   
    v_global = np.multiply(((x.transpose()-gbest).transpose()),r2)*c2*(-1.0)    # velocity towards global optima
    v_local = np.multiply((pbest- x),r1)*c1           # velocity towards local optima (pbest)

    v = w*v + v_local + v_global        # velocity update
  
    x = x + v                           # position update
    
    
    fit = fitness(x)                    # fitness function call (once per iteration). Vector Np
    
    # pbest and pbest_val update
    ind = np.argwhere(fit > pbest_val)  # indices where current fitness value set is greater than pbset
    pbest_val[ind] = np.copy(fit[ind])  # update pbset_val at those particle indices where fit > pbest_val
    pbest[:,ind] = np.copy(x[:,ind])    # update pbest for those particle indices where fit > pbest_val
  
    
    # gbest and gbest_val update
    ind2 = np.argmax(pbest_val)                       # index where the fitness is maximum
    gbest_val[iter] = np.copy(pbest_val[ind2])        # store gbest value at each iteration
    gbest = np.copy(pbest[:,ind2])                    # global best solution, gbest
    
    gbest_store[:,iter] = np.copy(gbest)              # store gbest solution

    print("Iter. =", iter, ". gbest_val = ", gbest_val[iter])  # print iteration no. and best solution at each iteration
    
    


# Plotting
py.close('all')
py.figure(1)
py.plot(gbest_val)
py.xlabel('iterations')
py.ylabel('fitness, gbest_val')
py.figure(2)
py.plot(gbest_store[1,:],'r')
py.xlabel('iterations')
py.ylabel('gbest[1,iter]')


        