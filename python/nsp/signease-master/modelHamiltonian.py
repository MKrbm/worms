# -*- coding: utf-8- -*-
""" model Hamiltonian 

The module provides a subclass of Hamiltonian implementing the concrete 
local Hamiltonian models that where numerically studied in the publication 

	Hangleiter, Roth, Nagaj, and Eisert. "Easing the Monte Carlo sign problem" 
	https://arxiv.org/abs/1906.02309

Authors: 
	Dominik Hangleiter and Ingo Roth


Licence: 
	This project is licensed under the MIT License - see the LICENSE.md file for details.
"""

from circuitOptimizer import *

class ModelHamiltonian(Hamiltonian):
	"""Class implementing a couple of model Hamiltonians.

	Attributes (inherited from `Hamiltonian`):
		locality (int): the locality of the local Hamiltonian terms
		localDim (int): local Hilbert space dimension. The local term is 
		a matrix of size `localDim`**`locality`.

	"""

	def __init__(self, model):
		"""Init method for model Hamiltonian.

		Args: 
			model (:obj: `dict`): Specifies the model. Allowed values for 
				'name' (string): name of the model. Allowed values:
					'j0j1j2j3',
					'j0j1j2j3orig',
					'j0j1j2j3left',
					'j0j1j2j3right',
					'frustratedLadder',
					'frustratedLadderDimer'.
				'pars': parameters of the model
				
				See the implementation of the called init function for each 
				model for detailed documentation. 

		Raises:
			ValueError: If  the model name is not known. 
		"""

		self.locality = model["locality"]
		self.localDim = model["localDim"]


		try:
			localTermFunction = getattr(self, model["name"])
		except: 
			raise ValueError('Hamiltonian model ' + model["name"] + ' not available.')

		self.localTerm = localTermFunction(model["pars"])


	def j0j1j2j3(self,pars):
		"""Define the Hamiltonian of the J0-J1-J2-J3-model.

		The Hamiltonian of the J0-J1-J2-J3-model as a function of the parameters, 
		where two spins are combined into a single one so that, effectively, 
		the model is one-dimensional. See Nakamura, Phys Rev B: 57, R3197 (1998),
		 in the singlet-triplet basis.

		Args: 
			pars (list type): Parameters of the model.

		Returns:
			:obj: `numpy.array`: The local hamlitonian matrix
	
		"""

		j0 = pars[0]
		j1 = pars[1]
		j2 = pars[2]
		j3 = pars[3]

		a = j0 + j1 + j3
		b = -j0 + j1 + j3
		c = j0 - j1 + j3
		d= -j0 -j1 + j3

		h1 = - np.array([[ -j2,-a, -a, -d], 
						 [-a, -j2, a, d], 
						 [ -a, a , -j2, d], 
						 [-d, d, d, 3 * j2]])

		h2 = - np.array([[ -j2,-a, -b, -c], 
						 [-a, -j2, b, c], 
						 [ -b, b , j2, d], 
						 [-c, c, d, j2]])

		H = sla.block_diag(h1, h2, h2, h2)

		return .25 * H 

	def j0j1j2j3left(self,pars):
		"""Define the Hamiltonian of the J0-J1-J2-J3-model.

		The Hamiltonian of the J0-J1-J2-J3-model as a function of the parameters, 
		where two spins are combined into a single one so that, effectively, 
		the model is one-dimensional. See Nakamura, Phys Rev B: 57, R3197 (1998), 
		in the computational basis placed to left.


		Args: 
			pars (list type): Parameters of the model.

		Returns:
			:obj: `numpy.array`: The local hamlitonian matrix

		"""
		j0 = pars[0]
		j1 = pars[1]
		j2 = pars[2]
		j3 = pars[3]

		Id = np.eye(2)
		Id2 = np.eye(4)

		X = np.array([[0,1]
					 ,[1,0]])

		Y = np.array([[0,-1j],
					  [1j,0]])

		Z = np.array([[1,0],
					 [0,-1]])
		
		X1 = np.kron(X,Id)
		X2 = np.kron(Id,X)

		Y1 = np.kron(Y,Id)
		Y2 = np.kron(Id,Y)
		
		Z1 = np.kron(Z,Id)
		Z2 = np.kron(Id,Z)

		orthTerm1 = np.kron(np.dot(X1,X2) + np.dot(Y1,Y2) + np.dot(Z1,Z2),Id2)
		orthTerm2 = np.kron(Id2,np.dot(X1,X2) + np.dot(Y1,Y2) + np.dot(Z1,Z2))
		
		crossTerm= np.kron(X2,X1) + np.kron(Y2,Y1) + np.kron(Z2,Z1)

		parTerm0 = np.kron(X1,X1) + np.kron(Y1,Y1) + np.kron(Z1,Z1)
		parTerm1 = np.kron(X2,X2) + np.kron(Y2,Y2) + np.kron(Z2,Z2)

		hamiltonianTerm = j0 *parTerm0 + j1*parTerm1 \
						+ j2 * orthTerm1  + j3* crossTerm

		return .25*np.real(hamiltonianTerm)


	def j0j1j2j3right(self, pars):
		"""Define the Hamiltonian of the J0-J1-J2-J3-model.

		The Hamiltonian of the J0-J1-J2-J3-model as a function of the parameters, 
		where two spins are combined into a single one so that, effectively, 
		the model is one-dimensional. See Nakamura, Phys Rev B: 57, R3197 (1998), 
		in the computational basis placed to the right.

		Args: 
			pars (list type): Parameters of the model.

		Returns:
			:obj: `numpy.array`: The local hamlitonian matrix

		"""

		j0 = pars[0]
		j1 = pars[1]
		j2 = pars[2]
		j3 = pars[3]

		Id = np.eye(2)
		Id2 = np.eye(4)
		X = np.array([[0,1],[1,0]])
		Y = np.array([[0,-1j],[1j,0]])
		Z = np.array([[1,0],[0,-1]])
		
		X1 = np.kron(X,Id)
		X2 = np.kron(Id,X)

		Y1 = np.kron(Y,Id)
		Y2 = np.kron(Id,Y)
		
		Z1 = np.kron(Z,Id)
		Z2 = np.kron(Id,Z)

		orthTerm1 = np.kron(np.dot(X1,X2) + np.dot(Y1,Y2) + np.dot(Z1,Z2),Id2)
		orthTerm2 = np.kron(Id2,np.dot(X1,X2) + np.dot(Y1,Y2) + np.dot(Z1,Z2))
		
		crossTerm= np.kron(X2,X1) + np.kron(Y2,Y1) + np.kron(Z2,Z1)

		parTerm0 = np.kron(X1,X1) + np.kron(Y1,Y1) + np.kron(Z1,Z1)
		parTerm1 = np.kron(X2,X2) + np.kron(Y2,Y2) + np.kron(Z2,Z2)

		hamiltonianTerm = j0 *parTerm0 + j1*parTerm1 \
						+ j2* orthTerm2 + j3* crossTerm

		return .25*np.real(hamiltonianTerm)


	def j0j1j2j3orig(self, pars):
		"""Define the Hamiltonian of the J0-J1-J2-J3-model.

		The Hamiltonian of the J0-J1-J2-J3-model as a function of the parameters, 
		where two spins are combined into a single one so that, effectively, 
		the model is one-dimensional. See Nakamura, Phys Rev B: 57, R3197 (1998), 
		in the computational basis placed in the center.

		Args: 
			pars (list type): Parameters of the model.

		Returns:
			:obj: `numpy.array`: The local hamlitonian matrix

		"""
		j0 = pars[0]
		j1 = pars[1]
		j2 = pars[2]
		j3 = pars[3]

		Id = np.eye(2)
		Id2 = np.eye(4)
		X = np.array([[0,1],[1,0]])
		Y = np.array([[0,-1j],[1j,0]])
		Z = np.array([[1,0],[0,-1]])
		
		X1 = np.kron(X,Id)
		X2 = np.kron(Id,X)

		Y1 = np.kron(Y,Id)
		Y2 = np.kron(Id,Y)
		
		Z1 = np.kron(Z,Id)
		Z2 = np.kron(Id,Z)

		orthTerm1 = np.kron(np.dot(X1,X2) + np.dot(Y1,Y2) + np.dot(Z1,Z2),Id2)
		orthTerm2 = np.kron(Id2,np.dot(X1,X2) + np.dot(Y1,Y2) + np.dot(Z1,Z2))
		
		crossTerm= np.kron(X2,X1) + np.kron(Y2,Y1) + np.kron(Z2,Z1)

		parTerm0 = np.kron(X1,X1) + np.kron(Y1,Y1) + np.kron(Z1,Z1)
		parTerm1 = np.kron(X2,X2) + np.kron(Y2,Y2) + np.kron(Z2,Z2)

		hamiltonianTerm = j0 *parTerm0 + j1*parTerm1 \
						+ j2/2 * (orthTerm1 + orthTerm2) + j3* crossTerm

		return .25*np.real(hamiltonianTerm)


	def frustratedLadder(self, pars):
		"""Define the Hamiltonian of a frustrated ladder model.


		The Hamiltonian of the frustrated ladder model discussed by 
		Wessel et al, SciPost Physics 3, 005 (2017) in the computational basis.


		Args: 
			pars (list type): Parameters of the model.

		Returns:
			:obj: `numpy.array`: The local hamlitonian matrix

		"""
		jOrth = pars[0]
		jParallel = pars[1]
		jCross = pars[2]

		Id = np.eye(2)
		Id2 = np.eye(4)
		X = np.array([[0,1],[1,0]])
		Y = np.array([[0,-1j],[1j,0]])
		Z = np.array([[1,0],[0,-1]])
		
		X1 = np.kron(X,Id)
		X2 = np.kron(Id,X)

		Y1 = np.kron(Y,Id)
		Y2 = np.kron(Id,Y)
		
		Z1 = np.kron(Z,Id)
		Z2 = np.kron(Id,Z)

		onSiteTermL = np.kron(np.dot(X1,X2) + np.dot(Y1,Y2) + np.dot(Z1,Z2),Id2)
		# onSiteTermR = np.kron(Id2,np.dot(X1,X2) + np.dot(Y1,Y2) + np.dot(Z1,Z2))

		crossTerm1 = np.kron(X1,X2) + np.kron(Y1,Y2) + np.kron(Z1,Z2)
		crossTerm2 = np.kron(X2,X1) + np.kron(Y2,Y1) + np.kron(Z2,Z1)

		parTerm1 = np.kron(X1,X1) + np.kron(Y1,Y1) + np.kron(Z1,Z1)
		parTerm2 = np.kron(X2,X2) + np.kron(Y2,Y2) + np.kron(Z2,Z2)

		hamiltonianTerm = jCross * ( crossTerm1 + crossTerm2) \
						+ jParallel * ( parTerm1 + parTerm2) \
						+  jOrth * onSiteTermL 
		# + onSiteTermR) + 

		return np.real(hamiltonianTerm)


	def frustratedLadderDimer(self, pars):
		"""Define the Hamiltonian of a frustrated ladder model in the dimer basis.


		The Hamiltonian of the frustrated ladder model in the dimer basis 
		as given by Wessel et al, SciPost, 2017.

		Args: 
			pars (list type): Parameters of the model.

		Returns:
			:obj: `numpy.array`: The local hamlitonian matrix

		"""

		jOrth = pars[0]
		jParallel = pars[1]
		jCross = pars[2]

		Id = np.eye(2)
		Id2 = np.eye(4)
		X = np.array([[0,1],[1,0]])
		Y = np.array([[0,-1j],[1j,0]])
		Z = np.array([[1,0],[0,-1]])
		
		X1 = np.kron(X,Id)
		X2 = np.kron(Id,X)

		Y1 = np.kron(Y,Id)
		Y2 = np.kron(Id,Y)
		
		Z1 = np.kron(Z,Id)
		Z2 = np.kron(Id,Z)

		TX = X1 + X2 
		TY = Y1 + Y2
		TZ = Z1 + Z2
		
		DX = X1 - X2 
		DY = Y1 - Y2
		DZ = Z1 - Z2

		onSiteTerm = np.kron(TX.dot(TX),Id2) \
					+ np.kron(TY.dot(TY),Id2) \
					+ np.kron(TZ.dot(TZ),Id2)

		tTerm = np.kron(TX,TX) + np.kron(TY,TY) + np.kron(TZ,TZ)
		dTerm = np.kron(DX,DX) + np.kron(DY,DY) + np.kron(DZ,DZ)
		
		hamiltonianTerm = .5 * (jCross + jParallel)* tTerm \
						+ .5*(jParallel - jCross) * dTerm  \
						+ jOrth*(.5 *  onSiteTerm - 3 * np.kron(Id2,Id2))

		return np.real(hamiltonianTerm)