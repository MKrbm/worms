# -*- coding: utf-8 -*-
"""Produce random stoquastic data

This script generates the data for the plots for randomly drawn stoquastic 
Hamiltonians using parallelisation. 

The script was used to create the plots in 

    Hangleiter, Roth, Nagaj, and Eisert. "Easing the Monte Carlo sign problem" 
    https://arxiv.org/abs/1906.02309

Authors: 
    Dominik Hangleiter and Ingo Roth

Licence:
	This project is licensed under the MIT License - see the LICENSE.md file for details.
"""

from f_optimisation import *

np.set_printoptions(precision=2)
verbose = True

# Set parameters of the problem 
locality = 2
localDim = 4

# Define the modelName 
# modelName = 'StoquasticGaussianProjected'
modelName = 'StoquasticHaarProjected'
# Random pars
modelPars = None

# Set optimization parameters
initialize = 'identity'
stepSizeFro =  'grid' # Set to 1E-3 for j0j1j2j3 model 
stepSizeL1 = 'grid' # set to 1E-5 #
max_steps = 500 
L1_cutoff = None # 1E-3 #
alpha = 50

hybrid = False

# Set average sign parameters
nSites = 0
nMC = 100
beta = 1.

# plot parameters
nSamples = 100
nDim = 5
resultsFinal = np.zeros((nSamples,nDim))
resultsInitial = np.zeros((nSamples,nDim))

for dd in np.arange(nDim) :
	localDim = dd + 2
	print(localDim)
	for kk in range(nSamples):
		currentCircuit, currentReport, objvals, _ = modelOptimizer(locality,localDim,modelName,modelPars,nSites,nMC,beta,initialize = initialize,stepSizeFro = stepSizeFro,stepSizeL1 = stepSizeL1,max_steps = max_steps,L1_cutoff = L1_cutoff,alpha = alpha, verbose = verbose,returnAvSign = False,doL1 = False)
		resultsFinal[kk,dd] = objvals[0] 
		resultsInitial[kk,dd] = objvals[1]
		# print('--------------------')
		# print("I am at iteration",kk)
		# print("and have just obtained",objvals)
	# results = {'hamtype':hamtype,'nqbit':nqbit,'samples':samples,'beta':beta,'nMC':nMC, 'alpha': nStoqPar, 'avsign': avsign}

results = {'afterOpt':resultsFinal,'beforeOpt':resultsInitial}

# path = '/net/storage/dhangleiter/sign_easing/data/'
path = 'data/'
savestring = 'modelH_'+modelName+'-init_'+initialize+'-locality_{}-localDim_{}-pars_{}-alpha_{}-L1Cutoff_{}-max_steps_{}-froStep_{}-l1Step_{}-hybrid_{}'.format(locality,localDim,modelPars,alpha,L1_cutoff,max_steps,stepSizeFro,stepSizeL1,hybrid)+'_3'

np.save(path+savestring,results)

plotpath = 'plots/'
savestringCSV = plotpath+savestring+'.csv'
plotData = (resultsFinal + 1E-15)/(resultsInitial + 1E-15) # Add tiny number to make results log-plottable 
np.savetxt(savestringCSV, plotData, delimiter = ', ')
