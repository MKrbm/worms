# -*- coding: utf-8- -*-
""" f_optimisation: optimization wrapper

The module provides a wrapper function that allows to access the functionality 
of the circuitOptimizer module for easing the Monte-Carlo sign problem 
for the models provided by the modelHamiltonian models and the simple 
Ansatz class of IdenticalOnsiteCircuit. It is used for the paralellization of 
the computation. 

The wrapper was used in the numerical study of

	Hangleiter, Roth, Nagaj, and Eisert. "Easing the Monte Carlo sign problem" 
	https://arxiv.org/abs/1906.02309

Authors: 
	Dominik Hangleiter and Ingo Roth


Licence: 
	This project is licensed under the MIT License - see the LICENSE.md file for details.
"""

from circuitOptimizer import *
from modelHamiltonian import *
from identicalOnsiteCircuit import *


# """
# Module level functions
# """

def modelOptimizer(locality, localDim, modelName, modelPars, nSites, nMC, beta,
					initialize = None, 
					stepSizeFro = 'grid', 
					stepSizeL1 = 'grid',
					max_steps = 500, 
					L1_cutoff = None, 
					alpha = 50, 
					verbose = True, 
					returnAvSign = True, 
					doL1 = True):
	"""Wrapper function to run the conjugate-gradient optimization for random 
		stoquastic Hamiltonians, and the frustrated spin ladders of 
		Nakamura Phys Rev B: 57, R3197 (1998) 
		and Wessel et al, SciPost Physics 3, 005 (2017)
		
		Args: 
			locality (int)		: 	locality of the Hamiltonian
			localDim (int)		: 	local dimension of the system
			modelName (string) 	: 	identifier of the model 
									- Random ensembles: 'stoquasticHaar', 
									'stoquasticGaussian', 'stoquasticProjectedHaar', 
									'stoquasticProjectedGaussian', 'PositiveHaar', 
									'RealGaussian'
									- 'j0j1j2j3' model of Nakamura et al
										and variants: 'j0j1j2j3orig', 'j0j1j2j3left', 
										'j0j1j2j3right'.
									- 'frustratedLadder' model of Wessel et al.
										and variant: 'frustratedLadderDimer'
			modelPars(array type or None)	:	parameters for the model or 
												`None` for random ensembles, 
												array for ladder models.
			nSites (int)	: 	number of sites on which the average sign is evaluated
			nMC (int)		: 	number of Monte Carlo steps for the average sign
			beta (float)	: 	inverse temperature for evaluation of the average sign
			initialize (string, optional)	: 	takes value `None- for initialization
											 at the identity, and 'random' 
											 for a Haar random circuit. 
			stepSizeFro	(string/float, optional): 	'conj' step size obtained by a polynomial fit
											'grid' step size obtained by a gridding procedure
											(float) constant step size 
			stepSizeL1	(string/float, optional): 	'conj' step size obtained by a polynomial fit
											'grid' step size obtained by a gridding procedure
											(float) constant step size 
			max_steps (int, optional)				: 	maximum number of steps in the optimization
			L1_cutoff (float/None, optional)	: Cutoff value for the gradient calculation 
											`None` for no cutoff
			alpha (float/None, optional)	: 	Smoothing parameter if smoothing 
										of the l1-norm is used, `None`
										for the usual l1 norm
			returnAvSign (logical, optional) 	: 	Calculate average sign or not
			hybrid (logical, optional)		: 	Perform l1 norm minimization 
										from scratch or not


		
		Returns: tuple of 
			currentCircuit (:obj: `Circuit`): the optimal on-site orthogonal basis 
			currentReport (:obj: `Dict`): dictionary of the results of the optimization
			objvals (:obj: `numpy.array`): [final value, intial value] of the objective function
			avsigns (:obj: `numpy.array`) [final value, intial value] of the average sign 
							for the given parameters

	"""


	# Initialize the measure at Frobenius norm
	myMeasure = Measure(localDim=localDim, measureType={'name':"fro"})

	# Define the Hamiltoniain 
	if (modelName in ['j0j1j2j3', 'j0j1j2j3orig', 'j0j1j2j3right', 'j0j1j2j3left', 
						'frustratedLadder', 'frustratedLadderDimer']):
		model = {"name" : modelName, 
				 "pars" : modelPars, 
				 "locality" : 2,
				 "localDim" : 4}

		H = ModelHamiltonian(model)
	elif (modelName in ['StoquasticGaussian', 'StoquasticHaar', 
				'StoquasticHaarProjected', 'StoquasticGaussianProjected']):
		H = RandHamiltonian(locality,localDim, 
							seed = modelPars, ensemble = modelName)
		randomCircuit = IdenticalOnsiteCircuit(localDim = localDim)
		randomCircuit.initRandom()
		H = randomCircuit.conjugationLocalHamiltonian(H)
	else:	
		H = RandHamiltonian(locality, localDim, 
							seed = modelPars, ensemble =modelName)

	# Initialize the Optimizer
	optimizer = Optimizer(myMeasure, H)

	# We still need the identity circuit for later use
	identityCircuit = IdenticalOnsiteCircuit(localDim=localDim)
	identityCircuit.coreOrths[0] = np.eye(localDim)
	
	# Initialize the circuit 
	if initialize == 'random':
		# Initialize the circuit randomly
		randomCircuit = IdenticalOnsiteCircuit(localDim=localDim)
		randomCircuit.initRandom()
		initCircuit = randomCircuit

	elif initialize == 'perturbedIdentityUncorrected':
		# Initialize the circuit close to the identity
		randomCircuit = IdenticalOnsiteCircuit(localDim=localDim)
		randomCircuit.initRandom()
		randomCircuit.coreOrths[0],_ = np.linalg.qr(np.eye(localDim) + .1*randomCircuit.coreOrths[0])
		
		initCircuit = randomCircuit

	elif initialize == 'perturbedIdentity':
		# Initialize the circuit close to the identity and correct for signs
		randomCircuit = IdenticalOnsiteCircuit(localDim=localDim)
		randomCircuit.initRandom()
		Q,R = np.linalg.qr(np.eye(localDim) + .1*randomCircuit.coreOrths[0])
		randomCircuit.coreOrths[0] = Q.dot(np.diag(np.sign(np.diag(R)))) #Correct for signs of diagonal entries of R 

		initCircuit = randomCircuit

	elif initialize == 'singlet-triplet':
		singTrip2 = 1/np.sqrt(2) * np.array([[0,0,1,1],
											 [1,1,0,0],
											 [-1,1,0,0],
											 [0,0,1,-1]])

		singTrip = 1/np.sqrt(2) * np.array([[1,0,0,1],
											[0,1,1,0],
											[0,-1,1,0],
											[1,0,0,-1]])

		initCircuit = IdenticalOnsiteCircuit(localDim = localDim)
		initCircuit.coreOrths[0] = singTrip

	else:
		# Initialize at the identity  
		initCircuit = identityCircuit

	optimizer.initCircuit = initCircuit
	print(initCircuit.coreOrths[0])

	print('Started Frobenius norm minimization')
	print('------------------------------------')
	# Run the Frobenius norm optimization 
	if stepSizeFro == 'grid':
		froCircuit, froReport = optimizer.optimize(
				verbose = verbose, stepper = 'grid', max_steps = max_steps)
	elif stepSizeFro == 'conj':
		froCircuit, froReport = optimizer.optimize(
				verbose = verbose, max_steps = max_steps)
	else:
		froCircuit, froReport = optimizer.optimize(
				verbose = verbose, stepper = 'constant', 
				givenStepSize = stepSizeFro, max_steps = max_steps)

	if froReport[0]['conv'] is False:
		warnings.warn('Frobenius norm optimization did not converge' )

	print('Started l1 norm minimization at the current point')
	print('------------------------------------')

	# Initialize the measure as the l1 measure
	if alpha is not None:
		optimizer.measure.measureType = {'name':'smooth-ell1','alpha':alpha}
	elif L1_cutoff is not None:
		optimizer.measure.measureType = {'name':'ell1','epsilon':L1_cutoff}
	else:
		raise ValueError('No valid l1-measure given.')

	# Initialize at the Frobenius measure optimum
	optimizer.initCircuit = froCircuit

	if stepSizeL1 == 'grid':
		l1Circuit, l1Report = optimizer.optimize(
				verbose = verbose, stepper = 'grid', max_steps=max_steps)
	elif stepSizeL1 == 'conj':
		l1Circuit, l1Report = optimizer.optimize(
				verbose = verbose, max_steps=max_steps)
	else:
		l1Circuit, l1Report = optimizer.optimize(
				verbose = verbose, stepper = 'constant', 
				givenStepSize = stepSizeL1, max_steps = max_steps)

	if l1Report[0]['conv'] is False:
		warnings.warn('l1-norm optimization did not converge' )

	if optimizer.objectiveFunction(froCircuit) < optimizer.objectiveFunction(l1Circuit):
		currentCircuit = froCircuit
		currentReport = froReport
		warnings.warn('Ohhhps: the l1-minimization made it worse')

	if doL1:
		print('Started l1-norm minimization from scratch')
		print('------------------------------------')

		# Run only L1-norm minimization
		if alpha is not None:
			optimizer.measure.measureType = {'name':'smooth-ell1','alpha':alpha}
		elif L1_cutoff is not None:
			optimizer.measure.measureType = {'name':'ell1','epsilon':L1_cutoff}
		else:
			raise ValueError('No valid l1-measure given.')

		# Initialize again from scratch 
		optimizer.initCircuit = initCircuit
		
		if stepSizeL1 == 'grid':
			onlyL1Circuit, onlyL1Report = optimizer.optimize(
					verbose = verbose, stepper = 'grid', max_steps=max_steps)
		elif stepSizeL1 == 'conj':
			onlyL1Circuit, onlyL1Report = optimizer.optimize(
					verbose = verbose, max_steps=max_steps)
		else:
			onlyL1Circuit, onlyL1Report = optimizer.optimize(
					verbose = verbose, stepper='constant', 
					givenStepSize = stepSizeL1, max_steps=max_steps)

		if l1Report[0]['conv'] is False:
			warnings.warn('only l1-norm optimization did not converge' )
	else:
		onlyL1Circuit = identityCircuit


	# Set the measure to the  L1 norm for evaluation of the optimization 

	optimizer.measure.measureType = {'name': 'ell1','epsilon':0}
	argmin = np.argmin([
			optimizer.objectiveFunction(froCircuit),
			optimizer.objectiveFunction(l1Circuit),
			optimizer.objectiveFunction(onlyL1Circuit)])
	if argmin == 0: 
		currentCircuit = froCircuit
		currentReport = froReport
	elif argmin == 1: 
		currentCircuit = l1Circuit
		currentReport = l1Report
	elif argmin == 2: 
		currentCircuit = onlyL1Circuit
		currentReport = onlyL1Report

	print('The argmin is', argmin)

	if verbose:

		# Evaluate the nonstoquasticity after Frobenius norm minimization
		print('Improvement after Frobenius norm minimization:')
		print(np.abs(optimizer.objectiveFunction(identityCircuit) \
						/(optimizer.objectiveFunction(froCircuit)+ 1E-15)))
		print('------------------------------------')
		# Evaluate the nonstoquasticity after L1 norm minimization
		print('Improvement after l1-norm minimization:')
		print(np.abs(optimizer.objectiveFunction(identityCircuit) \
						/(optimizer.objectiveFunction(l1Circuit) + 1E-15)))
		print('------------------------------------')
		if doL1:
			# Evaluate the nonstoquasticity after only l1 norm minimization
			print('Improvement after only l1-norm minimization:')
			print(np.abs(optimizer.objectiveFunction(identityCircuit) \
							/(optimizer.objectiveFunction(onlyL1Circuit) + 1E-15)))
			print('------------------------------------')

	# Return optimal Circuit, optimization reports, final and initial value of the objective function
	
	print('Initial value of the objective Function:')
	print(optimizer.objectiveFunction(identityCircuit))
	print('Final value of the objective Function:')
	print(optimizer.objectiveFunction(currentCircuit))
	print('------------------------------------')
	
	objvals = [ optimizer.objectiveFunction(currentCircuit), 
				optimizer.objectiveFunction(identityCircuit)]
	
	avsigns = []

	if returnAvSign:
		print('The original average sign ')
		print(identityCircuit.conjugationLocalHamiltonian(H).avSign(nSites,nMC,beta))
		print('The final average sign ')
		print(currentCircuit.conjugationLocalHamiltonian(H).avSign(nSites,nMC,beta))
		print('------------------------------------')

		avsigns = [currentCircuit.conjugationLocalHamiltonian(H).avSign(nSites,nMC,beta),
					identityCircuit.conjugationLocalHamiltonian(H).avSign(nSites,nMC,beta)]
		
	return currentCircuit, currentReport, objvals, avsigns


def saveModelOptimizer(locality, localDim, modelName, modelPars, nSites, nMC, 
						beta, initialize, stepSizeFro, stepSizeL1, max_steps, 
						L1_cutoff, alpha, hybrid):
	"""
		Capsulated function for carrying out optimization in a parallelized way and saving : 
		
		Args: 
			locality (int)		: 	locality of the Hamiltonian
			localDim (int)		: 	local dimension of the system
			modelName (string) 	: 	identifier of the model 
									- Random ensembles: 'stoquasticHaar', 
									'stoquasticGaussian', 'stoquasticProjectedHaar', 
									'stoquasticProjectedGaussian', 'PositiveHaar', 
									'RealGaussian'
									- 'j0j1j2j3' model of Nakamura et al
										and variants: 'j0j1j2j3orig', 'j0j1j2j3left', 
										'j0j1j2j3right'.
									- 'frustratedLadder' model of Wessel et al.
										and variant: 'frustratedLadderDimer'
			modelPars(array type or None)	:	parameters for the model or 
												`None` for random ensembles, 
												array for ladder models.
			nSites (int)	: 	number of sites on which the average sign is evaluated
			nMC (int)		: 	number of Monte Carlo steps for the average sign
			beta (float)	: 	inverse temperature for evaluation of the average sign
			initialize (string or None)	: 	takes value `None- for initialization
											 at the identity, and 'random' 
											 for a Haar random circuit. 
			stepSizeFro	(string / float): 	'conj' step size obtained by a polynomial fit
											'grid' step size obtained by a gridding procedure
											(float) constant step size 
			stepSizeL1	(string / float): 	'conj' step size obtained by a polynomial fit
											'grid' step size obtained by a gridding procedure
											(float) constant step size 
			max_steps (int)				: 	maximum number of steps in the optimization
			L1_cutoff (float / None)	: Cutoff value for the gradient calculation 
											`None` for no cutoff
			alpha (float / None)	: 	Smoothing parameter if smoothing 
										of the l1-norm is used, `None`
										for the usual l1 norm
			returnAvSign (logical) 	: 	Calculate average sign or not
			hybrid (logical)		: 	Perform l1 norm minimization 
										from scratch or not

		
		Writes to file: 
			:obj: `Dict` : Dictionary of the results and parameters in the data / directory
	"""


	currentCircuit, currentReport, objvals, avsigns =  modelOptimizer(
		locality, localDim, modelName, modelPars, nSites, nMC, beta, 
		initialize = initialize, 
		stepSizeFro = stepSizeFro, 
		stepSizeL1 = stepSizeL1, 
		max_steps = max_steps, 
		L1_cutoff = L1_cutoff, 
		alpha=alpha, 
		verbose=False, 
		returnAvSign=True, 
		doL1=hybrid)

	path = 'data/'

	results = {
		'parameter values':modelPars,
		'objvals':objvals,
		'avsigns':avsigns,
		'initialize': initialize,
		'stepSizeFro':stepSizeFro,
		'stepSizeL1':stepSizeL1,
		'max_steps':max_steps,
		'L1_cutoff':L1_cutoff,
		'alpha':alpha,
		'hybrid':hybrid,
		'nSites':nSites,
		'nMC':nMC,
		'beta':beta}

	savestring = 'modelH_'+modelName \
		+'-init_'+initialize \
		+'-locality_{}-localDim_{}-pars_{}-alpha_{}-L1Cutoff_{}-max_steps_{}-froStep_{}-l1Step_{}-hybrid_{}'.format(
				locality,localDim,modelPars,alpha,L1_cutoff,max_steps,stepSizeFro,stepSizeL1,hybrid)

	print('-------------')
	print('The optimal values are ')
	print(objvals)

	np.save(path+savestring,results)
