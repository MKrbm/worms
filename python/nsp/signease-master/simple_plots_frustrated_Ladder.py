# -*- coding: utf-8 -*-
"""Simple plots frustrated

This script generates the data and a small plot for the frustrated latter model 
to demonstrate the easing of the Monte Carlo Sign problem. 

It is meant as a simple example to show the functionality of  the `circuitOptimizer`
 module that was used for the numerical study done in 

	Hangleiter, Roth, Nagaj, and Eisert. "Easing the Monte Carlo sign problem" 
	https://arxiv.org/abs/1906.02309

Examples: 
	Just run the script 
		python simple_plots_frustrated_Ladder.py
	and wait until a plot appears. 


Authors: 
	Dominik Hangleiter and Ingo Roth

Licence:
	This project is licensed under the MIT License - see the LICENSE.md file for details.
"""

from circuitOptimizer import *
from identicalOnsiteCircuit import *
from modelHamiltonian import *

import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
verbose = True

# # Set parameters of the problem 
locality = 2
localDim = 4

modelName = 'frustratedLadder'
# frustratedLadderpars  [Jorth, Jpar, Jcross]

nSamples = 2	#Increase this number and wait longer to see a larger plot. 
Jpar = 1
JorthMax = 1.5
JcrossMax = 1.5

# Set optimization parameters
alpha = 40
max_steps = 500

# Set average sign parameters
nSites = 4
nMC = 100
beta = 1.

# Initialize the identity circuit
identityCircuit = IdenticalOnsiteCircuit(localDim=localDim)
identityCircuit.coreOrths[0] = np.eye(localDim)

# Initialize arrays
Jorth = (np.arange(nSamples) +1) * JorthMax/nSamples
Jcross = (np.arange(nSamples) +1) * JcrossMax/nSamples

nonStoqBefore = np.zeros(2*(nSamples,))
nonStoqAfter = np.zeros(2*(nSamples,))
avSignBefore = np.zeros(2*(nSamples,))
avSignAfter = np.zeros(2*(nSamples,))
argmin = np.zeros(2*(nSamples,))

for orthK in range(nSamples):
	for crossL in range(nSamples):

		modelPars = [Jorth[orthK],1,Jcross[crossL]]
		model = {"name": modelName, "pars":modelPars,"locality":2,"localDim":4}
		H = ModelHamiltonian(model)
			
		myMeasure = Measure(localDim=localDim, measureType={'name':"fro"})
		optimizer = Optimizer(myMeasure, H)

		initCircuit = IdenticalOnsiteCircuit(localDim=localDim)
		initCircuit.initRandom()

		optimizer.initCircuit = initCircuit

		print('Started Frobenius norm minimization')
		print('------------------------------------')
		froCircuit, froReport = optimizer.optimize()

		if froReport[0]['conv'] is False:
			warnings.warn('Frobenius norm optimization did not converge' )


		# Run the subsequent L1-norm minimization
		print('Started l1 norm minimization at the current point')
		print('------------------------------------')
		myMeasure.measureType = {'name':'smooth-ell1','alpha':alpha}
		optimizer.initCircuit = froCircuit

		l1Circuit, l1Report = optimizer.optimize(max_steps = max_steps)

		# l1Circuit, l1Report = optimizer.optimize(verbose = verbose,stepper='constant',givenStepSize = stepSizeL1,max_steps=max_steps,fct_goal = fct_goal)

		if l1Report[0]['conv'] is False:
			warnings.warn('l1-norm optimization did not converge' )

		if optimizer.objectiveFunction(froCircuit) < optimizer.objectiveFunction(l1Circuit):
			currentCircuit = froCircuit
			currentReport = froReport
			warnings.warn('Ohhhps: the l1-minimization made it worse')

		print('Started l1-norm minimization from scratch')
		print('------------------------------------')
		myMeasure.measureType = {'name':'smooth-ell1','alpha':alpha}

		optimizer.initCircuit = initCircuit
			
		onlyL1Circuit, onlyL1Report = optimizer.optimize(max_steps = max_steps)

		if onlyL1Report[0]['conv'] is False:
			warnings.warn('only l1-norm optimization did not converge' )


		# Set the measure to the hard L1 norm for evaluation of the optimization 
		optimizer.measure.measureType = {'name': 'ell1','epsilon':0}
		argmin[orthK,crossL] = np.argmin([optimizer.objectiveFunction(froCircuit),optimizer.objectiveFunction(l1Circuit),optimizer.objectiveFunction(onlyL1Circuit)])
		if argmin.any() == 0: 
			currentCircuit = froCircuit
			currentReport = froReport
		elif argmin.any() == 1: 
			currentCircuit = l1Circuit
			currentReport = l1Report
		elif argmin.any() == 2: 
			currentCircuit = onlyL1Circuit
			currentReport = onlyL1Report

		print('The argmin is', argmin)
		
		nonStoqBefore[orthK,crossL] = optimizer.objectiveFunction(identityCircuit)
		nonStoqAfter[orthK,crossL] = optimizer.objectiveFunction(currentCircuit)
		nonStoq = nonStoqAfter/nonStoqBefore

		avSignBefore[orthK,crossL]= -np.log(identityCircuit.conjugationLocalHamiltonian(H).avSign(nSites,nMC,beta))
		avSignAfter[orthK,crossL] = - np.log(currentCircuit.conjugationLocalHamiltonian(H).avSign(nSites,nMC,beta))
		avSignRatio = avSignAfter/avSignBefore
		
		print('Initial value of the objective Function:')
		print(nonStoqBefore[orthK,crossL])
		print('Final value of the objective Function:')
		print(nonStoqAfter[orthK,crossL] )
		print('------------------------------------')
		print('The logarithm of the original inverse average sign ')
		print(avSignBefore[orthK,crossL])
		print('The logarithm of the final inverse average sign ')
		print(avSignAfter[orthK,crossL])
		print('------------------------------------')


		

print("Plotting ... Please wait ...")

plt.rc('text', usetex=True)
# plt.rc('font', family='Helvetica')
# Creates two subplots and unpacks the output array immediately
f, ax = plt.subplots(2, 2)

im1 = ax[0,0].imshow(nonStoq,extent=[JorthMax/nSamples,JorthMax,JcrossMax,JcrossMax/nSamples,])
ax[0,0].invert_yaxis()
plt.colorbar(im1,ax = ax[0,0])

im2 = ax[0,1].imshow(avSignRatio,extent=[JorthMax/nSamples,JorthMax,JcrossMax,JcrossMax/nSamples])
ax[0,1].invert_yaxis()
plt.colorbar(im2,ax = ax[0,1])

im3 = ax[1,0].imshow(avSignBefore,extent=[JorthMax/nSamples,JorthMax,JcrossMax,JcrossMax/nSamples])
ax[1,0].invert_yaxis()
plt.colorbar(im2,ax = ax[1,0])

im4 = ax[1,1].imshow(avSignAfter,extent=[JorthMax/nSamples,JorthMax,JcrossMax,JcrossMax/nSamples])
ax[1,1].invert_yaxis()
plt.colorbar(im4,ax = ax[1,1])


ax[1,0].set_xlabel(r"$J_\times/J_\parallel$")
ax[0,0].set_ylabel(r"$J_\perp/J_\parallel$")

ax[1,0].set_ylabel(r'$J_\perp/J_\parallel$')
ax[1,1].set_xlabel(r'$J_\times/J_\parallel$')

ax[0,0].set_title('Nonstoquasticity improvement')
ax[0,1].set_title(r'$\log \langle \mathrm{sign} \rangle ^{-1}$ improvement')
ax[1,0].set_title(r'$\log \langle \mathrm{sign} \rangle ^{-1}$ before optimization')
ax[1,1].set_title(r'$\log \langle \mathrm{sign} \rangle ^{-1}$ after optimization')
plt.tight_layout()

plt.savefig('plots/simple_plot_frustratedLadder.pdf', bbox_inches='tight')
print("Plot also saved to: \n\tplots/simple_plot_frustratedLadder.pdf")
plt.show()

