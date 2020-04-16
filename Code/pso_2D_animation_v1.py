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

from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})


# Control parameters
w = 0.5                   # Intertial weight
c1 = 2.0                  # Weight of searching based on the optima found by a particle
c2 = 2.0                  # Weight of searching based on the optima found by the swarm
v_fct = 0.02


Np = 60                   # population size (number of particles)
D = 2                     # dimension (= no. of parameters in the fitness function)
max_iter = 200            # maximum number of iterations 
xL = np.zeros(D) - 15      # lower bound (does not need to be homogeneous)  
xU = np.zeros(D) + 15      # upper bound (does not need to be homogeneous)   

delta = 0.025
xplt = yplt = np.arange(xL[0], xU[0], delta)
X, Y = np.meshgrid(xplt, yplt)
V1 = np.exp(-3*np.square(.3*Y-.9) - 3*np.square(.3*X-1.2))
V2 = np.exp(-4*np.square(.3*Y-2.1) - 4*np.square(.3*X+1.2))
V3 = np.exp(-2*np.square(.25*Y+1) - 2*np.square(.25*X+2))
V4 = np.exp(-2*np.square(.5*Y-1.2) - 2*np.square(.5*X-2.4))
V6 = np.exp(-2*np.square(.2*Y+1.4) - 4*np.square(.2*X-1.2))


V5 = np.sin(np.square(.1*X)) + np.sin(np.square(.15*Y))


F_plt = 1.4*V1 + 0.6*V2 + 0.9*V3 - V4 + 0.05*V5 + 0.8*V6

# Fitness function. The code maximizes the value of the fitness function
def fitness(x):
    # x is a matrix of size D x Np
    # The position of the entire swarmp is inputted at once. 
    # Thus, one function call evaluates the fitness value of the entire swarm
    # F is a vector of size Np. Each element represents the fitness value of each particle in the swarm
    
    V1 = np.exp(-3*np.square(.3*x[1,:]-.9) - 3*np.square(.3*x[0,:]-1.2))
    V2 = np.exp(-4*np.square(.3*x[1,:]-2.1) - 4*np.square(.3*x[0,:]+1.2))
    V3 = np.exp(-2*np.square(.25*x[1,:]+1) - 2*np.square(.25*x[0,:]+2))
    V4 = np.exp(-2*np.square(.5*x[1,:]-1.2) - 2*np.square(.5*x[0,:]-2.4))
    V5 = np.sin(np.square(.13*x[0,:])) + np.sin(np.square(.15*x[1,:]))
    V6 = np.exp(-2*np.square(.2*x[1,:]+1.4) - 4*np.square(.2*x[0,:]-1.2))

    # multimodal test function
    F_mult = 1.4*V1 + 0.6*V2 + 0.9*V3 - V4 +0.05*V5 + 0.8*V6
    
    
    return F_mult



pbest_val = np.zeros(Np)            # Personal best fintess value. One pbest value per particle.
gbest_val = np.zeros(max_iter)      # Global best fintess value. One gbest value per iteration (stored).


pbest = np.zeros((D,Np))            # pbest solution
gbest = np.zeros(D)                 # gbest solution

gbest_store = np.zeros((D,max_iter))   # storing gbest solution at each iteration

x = np.random.rand(D,Np)            # Initial position of the particles
v = np.zeros((D,Np))                # Initial velocity of the particles

x_store = np.zeros((max_iter,D,Np))




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


x_store[0,:,:] = x


# Loop over the generations
for iter in range(1,max_iter):
    
  
    r1 = np.random.rand(D,Np)           # random numbers [0,1], matrix D x Np
    r2 = np.random.rand(D,Np)           # random numbers [0,1], matrix D x Np   
    v_global = np.multiply(((x.transpose()-gbest).transpose()),r2)*c2*(-1.0)    # velocity towards global optima
    v_local = np.multiply((pbest- x),r1)*c1           # velocity towards local optima (pbest)

    v = w*v + v_local + v_global        # velocity update
  
    x = x + v*v_fct                     # position update
    
    
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
    x_store[iter,:,:] = x
    


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

#py.figure()
#py.plot(x_store[0,0,:],x_store[0,1,:],'o')
#py.xlim(xL[0],xU[0])
#py.ylim(xL[1],xU[1])
#
#py.figure()
#py.plot(x_store[max_iter-1,0,:],x_store[max_iter-1,1,:],'o')
#py.xlim(xL[0],xU[0])
#py.ylim(xL[1],xU[1])






fig = plt.figure()


ax = plt.axes(xlim=(xL[0], xU[0]), ylim=(xL[1],xU[1]),xlabel = 'x', ylabel= 'y')
#line, = ax.plot([], [], lw=2,,markersize = 9, markerfacecolor = "#FDB813",markeredgecolor ="#FD7813")
line1, = ax.plot([], [], 'o',color = '#d2eeff',markersize = 3, markerfacecolor = '#d2eeff',lw=0.1,  markeredgecolor = '#0077BE')   # line for Earth
time_template = 'Iteration = %i'
time_string = ax.text(0.05, 0.9, '', transform=ax.transAxes,color='white')



#ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
#ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
# initialization function: plot the background of each frame
def init():
    line1.set_data([], [])
    im2 = ax.contourf(X,Y,F_plt,40, cmap = 'plasma')
    
#    fig.colorbar(im2, ticks=[0, .3, .6, 1])
    time_string.set_text('')

    
    return line1, time_string

# animation function.  This is called sequentially
def animate(i):
    # Motion trail sizes. Defined in terms of indices. Length will vary with the time step, dt. E.g. 5 indices will span a lower distance if the time step is reduced.
    
    line1.set_data(x_store[i,0,:], x_store[i,1,:])   # marker + line of first weight
#    contourf(X,Y,F_plt)
    time_string.set_text(time_template % (i))
    return line1, time_string


v_fps = 20.0
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=max_iter, interval=1000.0/v_fps, blit=True)


anim.save('pso_slow_animation.mp4', fps=v_fps, extra_args=['-vcodec', 'libx264'])

        