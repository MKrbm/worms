# -*- coding: utf-8 -*-
"""Produce frustrated Ladder data

This script generates the data for the plots for the frustrated Ladder model 
using parallelisation. 

The script was used to create the plots in 

	Hangleiter, Roth, Nagaj, and Eisert. "Easing the Monte Carlo sign problem" 
	https://arxiv.org/abs/1906.02309

Authors: 
	Dominik Hangleiter and Ingo Roth

Licence:
    This project is licensed under the MIT License - see the LICENSE.md file for details.
"""

import numpy as np
from numpy import random as rand
from multiprocessing import Process, Pipe, Pool, Queue
from itertools import product

from f_optimisation import *

verbose = [False]

# Set parameters of the problem 
locality = [2]
localDim = [4]

# Define the model 

modelName = ['frustratedLadder']

nSamples = 21
JorthMax = 1.5
JcrossMax = 1.5

Jpar = [1]
Jorth = np.arange(nSamples)* JorthMax/(nSamples-1)
Jcross = np.arange(nSamples)* JcrossMax/(nSamples-1)

modelPars = product(Jorth,Jpar,Jcross)


# Set optimization parameters
initialize = ['perturbedIdentityUncorrected']
stepSizeFro =  ['grid']
stepSizeL1 = ['grid'] 
max_steps = [2000]
L1_cutoff =  [None]
alpha = [40] 
hybrid = [True]


# Set average sign parameters
nSites = [4]
nMC = [100]
beta = [1.]


# Define the parallelization routine 
class Worker(Process):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue= queue
    def run(self):
        print('Worker started')
        for data in iter(self.queue.get,None):
            print(data)
            saveModelOptimizer(*data)


request_queue = Queue()

for i in range(12):
    Worker(request_queue).start()

for data in product(locality,localDim,modelName,modelPars,nSites,nMC,beta,initialize,stepSizeFro,stepSizeL1,max_steps,L1_cutoff,alpha,hybrid):
    request_queue.put(data)

for i in range(12):
    request_queue.put(None)

