# -*- coding: utf-8 -*-
"""circuit optimizer

This module provides the basic classes to optimise a non-stoquasiticity measure 
over a set of quantum circuit acting on a local Hamiltonians. The implementation was 
used in the numerical simulation of the publication

    Hangleiter, Roth, Nagaj, and Eisert. "Easing the Monte Carlo sign problem" 
    https://arxiv.org/abs/1906.02309

The most important class types it provides are the follwing:
    * The class `Measure` implementing a couple of non-stoquasticity measures.
    * The type `Hamiltonian` implements the blue print of a local Hamiltonian model. 
        Specific models should be implement as childrens of the `Hamiltonian` class. 
    * The type `Circuit` specifies the blue print of a local quantum circuit consisting of multiple Orthogonal matrices.
        A specific circuit should implement as a subclass of `Circuit`.
    * Instances of the `Optimizer` class can perform a conjugate gradient optimisation of 
        an instace of `Measure` for a given Hamiltonian and circuit model, specified by 
        instances of `Hamiltonian` and `Circuit`.


The optimisation routine is based on the algorithm of 
    Abrudan, Traian, Jan Eriksson, and Visa Koivunen. 
    "Conjugate Gradient Algorithm for Optimization under Unitary Matrix Constraint."
    Signal Processing 89, no. 9 (September 2009): 1704–14. 
    https://doi.org/10.1016/j.sigpro.2009.03.015.


Authors: 
    Dominik Hangleiter and Ingo Roth

The implementation of the optimisation routine is based on code grateously provided 
by Christian Krumnow. 

Licence: 
    This project is licensed under the MIT License - see the LICENSE.md file for details.



Examples:
    A simple example use case can be found in 
        simple_plots_frustrated_Ladder.py
    Run and wait for some time ...
        python simple_plots_frustrated_Ladder.py

    This should display some plots.

    Executable examples as presented in the publication are:
        produce_j0model_data.py
        produce_randomStoquastic_data.py
        produce_frustratedLadder_data.py

    These scripts use the wrapper
        f_optimisation.py
    to access the functionality provided by the circuit optimiser module
    and use parallelisation to produce the data for the plots in the 
    publication. With standard desktop computing power theses script run 
    for about two hours.

    Subsequently, running the corresponding plot sripts 
        plotscript_j0model.py
        plotscript_randomStoquastic.py
        plotscript_frustratedLadder_full.py
    generates the plot displayed in the publication. 

TO DO: 
    * Move On-site orthogonal circuit to different file 
    * add licence
    * Truncate characters per line

CHANGE LOG:
    * changed adjunction to conjugation 
    * Moved hamiltonians models to seperate file

"""

import numpy as np
import warnings
import scipy.linalg as sla
import copy
from itertools import product 
import matplotlib.pyplot as plt


class Hamiltonian:
    """The class implements a translation-invariant, real Hamiltonian with period 1 in 1D.
        
    Attributes:
        locality (int): the locality of the local Hamiltonian terms
        localDim (int): local Hilbert space dimension. The local term is a matrix of size `localDim`**`locality`.

    """

    def __init__(self, locality=0, localDim=1, aMatrix=None): 
        """Init method for the Hamiltonion. 
        
        `localTerm` is set to `aMatrix` if provided otherwise initalised 
        as a zero matrix of given `localDim` to the power of `locality`.

        Args:
            locality (:obj: `integer`, optional): locality of the local Hamiltonian terms
            localDim (:obj: `integer`, optional): the local Dimension of the Hilbert space of the local hamiltionian.
            aMatrix (:obj: `numpy.array`. optional): initial values for the Hamiltonian local term.

        """

        if not aMatrix is None: 
            self.localTerm = aMatrix
        else: 
            self.localTerm = np.zeros((localDim**locality,localDim**locality))

        self.locality = locality
        self.localDim = localDim

    @property
    def localTerm(self):
        """:obj: `numpy.array`: Matrix representation of the local Hamiltonian term."""
        return self.__localTerm
    @localTerm.setter
    def localTerm(self, gMatrix):
        #if not checkHermitian(gMatrix):
        #	warnings.warn('Given matrix is not Hermitian!')
        self.__localTerm = gMatrix


    @property
    def totalDimension(self):
        """int: Returns the totalDimension of the local Hamiltonian term."""
        return self.localDim**self.locality

    def dot(self, gMatrix):
        """Wrapper for Hamiltonian.localTerm.dot

        Args:
            gMatrix (numpy.array): Argument for the dot function.

        Returns:
            numpy.array: result of matrix / vector multiplication 
                of self.localTerm with gMatrix.
        """
        return(self.localTerm.dot(gMatrix))

    def dotr(self, gMatrix):
        """Multiplies a given Matrix `gMatrix` from the right with the Hamiltonian term.

        Args:
            gMatrix (numpy.array): Argument for the dot function. 

        Returns:
            numpy.array: result of matrix / vector multiplication 
                of `gMatix` with `self.localTerm`
        """
        return(gMatrix.dot(self.localTerm))

    def translatedTerm(self,i,s):
        """Returns the matrix representation of the local Hamiltonian term action on the local site `i`of a system with `s` sites.
        Args:
            i (int): position of the local Termm
            s (int): total number of sites

        Returns:
            :obj: numpy.array: matrix form of the Kronecker product Id x ...x Id x `self.localTerm` x Id x ... x Id
        """

        reshapeIndices = 2 * s * (self.localDim,)
        return np.kron(np.kron(np.eye(self.localDim**i),self.localTerm),np.eye(self.localDim**(s - i - self.locality))).reshape(reshapeIndices)


    def avSign(self,nSites,nMC=100,beta= 1.):
        """ Returns the average sign of the Hamiltonian on `nSites` 
        with periodic boundary conditions and given Monte Carlo parameters.

        Args:
            nSites (int): Number of sites on which the Hamiltonian acts
            nMC (:obj: `int`, optional): Number of Monte Carlo steps
            beta (:obj: `float`, optional): Inverse Temperature 

        Returns:
            float: The average sign 

        """


        hilbertSpaceDim = self.localDim**nSites

        # Define the global Hamiltonian

        globalH = np.zeros((hilbertSpaceDim,hilbertSpaceDim))

        for i in range(nSites - 1): 
            globalH = globalH + self.translatedTerm(i,nSites).reshape(2*(hilbertSpaceDim,))

        T = np.eye(hilbertSpaceDim) - beta/nMC * globalH

        avsign = np.trace(np.linalg.matrix_power(T,nMC))/np.trace(np.linalg.matrix_power(abs(T),nMC))

        return avsign


class RandHamiltonian(Hamiltonian): 
    """Class of type Hamiltonian  that implements multiple random models for the local term.

    """
    def __init__(self, locality=0, localDim=1, ensemble='RealGaussian',seed = None):
        """Init method to draw random Hamiltonion from the given `ensemble`.
        

        Args:
            locality (:obj: `integer`, optional): locality of the local Hamiltonian terms
            localDim (:obj: `integer`, optional): the local Dimension of the Hilbert space of the local hamiltionian.
            ensemble (:obj: `string`, optional): The random ensemble. Allowed values are 
                'RealGaussian',
                'SparseGaussian',
                'positiveHaar',
                'SparsePositiveHaar',
                'StoquasticGaussian',
                'StoquasticGaussianProjected',
                'StoquasticHaar',
                'StoquasticHaarProjected'.

                All matrices are subsequently projected to Hermitian matrices.
            seed (:obj: `integer`, optional): seed for random generator passed to numpy.random.seed`
        

        Raises:
            ValueError: If string given as ensemble is not known. 
        """
        self.locality = locality
        self.localDim = localDim

        np.random.seed(seed)

        if ensemble=='RealGaussian': 
            randMatrix = np.random.randn(localDim**locality,localDim**locality)
        elif ensemble=='SparseGaussian': 
            randMatrix = np.random.randn(localDim**locality,localDim**locality)
            
            indexL = np.random.randint(localDim**locality,size = int(localDim**(2 * locality) - 2 * localDim**locality))
            indexR = np.random.randint(localDim**locality,size = int(localDim**(2 * locality) - 2 * localDim**locality))
            randMatrix[indexL,indexR] = 0 

        elif ensemble=='positiveHaar': 
            eigvals = np.random.uniform(0, 1, size=localDim**locality)
            
            Q,R = np.linalg.qr(np.random.randn(localDim**locality,localDim**locality))
            orth = Q.dot(np.diag(np.sign(np.diag(R)))) #Correct for signs of diagonal entries of R 
            
            randMatrix = orth.dot(np.diag(eigvals)).dot(orth.conj().T)

        elif ensemble=='SparsePositiveHaar': 
            eigvals = np.random.uniform(0, 1, size=localDim**locality)
            
            Q,R = np.linalg.qr(np.random.randn(localDim**locality,localDim**locality))
            orth = Q.dot(np.diag(np.sign(np.diag(R)))) #Correct for signs of diagonal entries of R 
            
            randMatrix = orth.dot(np.diag(eigvals)).dot(orth.conj().T)
            
            indexL = np.random.randint(localDim**locality,size = int(localDim**(2 * locality) - 2 * localDim**locality))
            indexR = np.random.randint(localDim**locality,size = int(localDim**(2 * locality) - 2 * localDim**locality))
            randMatrix[indexL,indexR] = 0 

        elif ensemble == 'StoquasticGaussian':
            randMatrix = np.random.randn(localDim**locality,localDim**locality)
            randMatrix = -np.abs(randMatrix)

        elif ensemble == 'StoquasticGaussianProjected':
            randMatrix = np.random.randn(localDim**locality,localDim**locality)
            randMatrixOffDiag = randMatrix - np.diag(np.diag(randMatrix))
            randMatrix = np.diag(np.diag(randMatrix)) + .5 * (randMatrixOffDiag -np.abs(randMatrixOffDiag))

        elif ensemble == 'StoquasticHaar':
            eigvals = np.random.uniform(0, 1, size=localDim**locality)
            
            Q,R = np.linalg.qr(np.random.randn(localDim**locality,localDim**locality))
            orth = Q.dot(np.diag(np.sign(np.diag(R)))) #Correct for signs of diagonal entries of R 
            
            randMatrix = orth.dot(np.diag(eigvals)).dot(orth.conj().T)
            randMatrix = -np.abs(randMatrix)

        elif ensemble == 'StoquasticHaarProjected':
            eigvals = np.random.uniform(0, 1, size=localDim**locality)
            orth, _ = np.linalg.qr(np.random.randn(localDim**locality,localDim**locality))
            randMatrix = orth.dot(np.diag(eigvals)).dot(orth.conj().T)

            randMatrixOffDiag = randMatrix - np.diag(np.diag(randMatrix))
            randMatrix = np.diag(np.diag(randMatrix)) + .5 * (randMatrixOffDiag - np.abs(randMatrixOffDiag))

        else:
            raise ValueError('Random ensemble '+ensemble+' is not implemented.')
        
        self.localTerm = .5*(randMatrix + randMatrix.T)






class Measure:
    """Implements measure on local Hamiltonians of type `Hamiltonian` that are optimised using the `Optimzer`. 

    The class implements different types and families of measures specified in 
    `self.measureType` as a dictionary. 

    Note:
        Currently the measures are only implemented for locality 2.

    Attributes:
        measureType (:obj: `dict`, optional): Specifies the measure that is evaluated. 
            Available measures and parameters
                {'name': 'smooth-ell1', 'alpha': int}
                    Smooth version of the l1 norm with smoothening parameter alpha. Approximates 
                    the l1-norm for large values of 'alpha'. 
                    Note: Values of alpha should not exceed 1E3 to avoid expensive evaluations. 
                {'name': 'ell1'} Standard l1 norm.
                {'name': 'fro'} Stanard frobenius / l2 norm
        _summationIndexMask (:obj: `tuple`): Precomputation of the indices selecting the relevant entries 
            for evaluation of the measure for locality 2. 

    """

    def __init__(self, locality=2, localDim=2, measureType={'name': "smooth-ell1", 'alpha': 1E2}):
        """Init method of Measure class. 

        Args:
            locality (:obj: integer, optional): initial value for the `locality` property.
            localDim (:obj: integer, optional): initial value for the local dimension property.
            measureType (:obj: dict, optional): specifies the measure that is evaluated. See class description for allowed values.
        """
        self.localDim = localDim
        self.locality = locality 
        self.measureType = measureType
        
        


    
    @property
    def locality(self):
        """The locality used in the evaluation of the measure. Default value is 2.

            Note:
                Currently the measures are only implemented for locality 2.
        """
        return(self._locality if hasattr(self, '_locality') else 2)
    @locality.setter
    def locality(self, givenLocality): 
        if givenLocality != self.locality or not hasattr(self, '_summationIndexMask'): 
            if givenLocality != 2: 
                raise ValueError('Objective function not implemented for locality '+givenLocality+' yet.') 
            self._locality = givenLocality
            self._summationIndexMask = self.calculateSummationIndexMask()
        
    @property
    def localDim(self):
        """The local dimension of the Hamiltonian for which the measure is evaluated.locality

            `_sumattionIndexMask` is computed whenever the value of `localDim` changes. 
        """
        return(self._localDim if hasattr(self, '_localDim') else 2)
    @localDim.setter
    def localDim(self, givenLocalDim): 
        if givenLocalDim != self.localDim or not hasattr(self, '_summationIndexMask'): 
            self._localDim = givenLocalDim
            self._summationIndexMask = self.calculateSummationIndexMask()
    
    

    def calculateSummationIndexMask(self):
        """Computes the indices relevant to the evaluation of the measure and gradient for locality 2. 

            Returns: 
                tuple: Tuple of indices addressing the matrix entries.
        """

        allIndicesMask = np.ones((self.localDim, self.localDim))
        offDiagIndicesMask = np.logical_not(np.eye(self.localDim))
        diagIndicesMask = np.eye(self.localDim)
        indices = np.nonzero(np.logical_not(np.kron(np.kron(allIndicesMask, offDiagIndicesMask), diagIndicesMask)))
        return indices

        

    def evaluate(self, hamiltonian, measure=None):
        """Returns the value of the measure for given `hamiltonian`.

        Args:
            hamiltonian (:obj: `Hamiltonian`): The hamiltonian for which the measure is computed. 
            measure (:obj: `dict`, optional): The measure that is computed. See class description for allowed values. 
                If not specified `self.measureType` is used. 

        Returns:
            float: value of the measure

        Raises:
            ValueError: If `measure` does not specify an implemented measure or the hamiltonian locality is not 2.
        """
        if measure is None:
            measure = self.measureType



        if measure['name'] == 'smooth-ell1':
            alpha = measure['alpha']
            f = lambda x: x + 1/alpha * np.log(1 + np.exp(-alpha *x))
        elif measure['name'] == 'ell1':
            f = lambda x: np.maximum(x,0)
        elif measure['name'] == 'fro' : 
            f = lambda x: np.maximum(x,0)**2
        else:
            raise ValueError('Type '+measure+' for the objective function not implemented.') 

        if hamiltonian.locality == 2:
            objectiveTerm = f(hamiltonian.translatedTerm(0,3) +  hamiltonian.translatedTerm(1,3))
            value = np.einsum('ikmjlm->',objectiveTerm) - np.einsum('ikmjkm->',objectiveTerm)

            value = value

        else:
            raise ValueError('Objective function not implemented for locality '+hamiltonian.locality+' yet.') 

        return value 


    def gradient(self,hamiltonian, measure=None):
        """Returns the gradient matrix of the measure at the `hamiltonian`.

        Args:
            hamiltonian (:obj: `Hamiltonian`): The hamiltonian for which the measure is computed. 
            measure (:obj: `dict`, optional): The measure that is computed. See class description for allowed values. 
                If not specified `self.measureType` is used. 

        Returns:
            numpy.array: The gradient matrix

        Raises:
            ValueError: If `measure` does not specify an implemented measure.
        """

        if measure is None:
            measure = self.measureType

        self.locality = hamiltonian.locality
        self.localDim = hamiltonian.localDim


        if measure['name'] == 'smooth-ell1':
            alpha = measure['alpha']
            grad_f = lambda x: 1 - 1/(1 + np.exp(alpha * x)) 
        elif measure['name'] == 'ell1':
            grad_f = lambda x: 1 * ( x >  measure['epsilon'] ) 
        elif measure['name'] == 'fro' :
            grad_f = lambda x: 2 * np.maximum(x,0)
        else:
            raise ValueError('Type '+measure+' for the objective function not implemented.') 

        #translateMatrix(0,3,hamiltonian.locality,hamiltonian.localTerm) +  translateMatrix(1,3,hamiltonian.locality,hamiltonian.localTerm) is adding two local hamiltonian.
        # H = h \otimes I + I \otimes h. where h is bond operator.
        outerDev = grad_f(translateMatrix(0,3,hamiltonian.locality,hamiltonian.localTerm) +  translateMatrix(1,3,hamiltonian.locality,hamiltonian.localTerm)) 
        test_ = grad_f(translateMatrix(1,3,hamiltonian.locality, hamiltonian.localTerm))
        # test = grad_f(translateMatrix(0,3,hamiltonian.locality,hamiltonian.localTerm))
        #reshape into matrix form hamiltonian from tensor form
        outerDev = outerDev.reshape(2*(self.localDim**(self.locality+1),))
        test_ = test_.reshape(2*(self.localDim**(self.locality+1),))
        # mask diagonal part.
        outerDev[self._summationIndexMask] = 0
        test_[self._summationIndexMask] = 0
        # retransform into tensor from matrix.
        outerDev = outerDev.reshape(2*(self.locality+1)*(self.localDim,))
        test_ = test_.reshape(2*(self.locality+1)*(self.localDim,))
        

        gradient = np.einsum("ijkimn", outerDev) + np.einsum("jkimni", outerDev)
        # grad_test = np.einsum("ijkimn", test_)

        return gradient.reshape(2*(self.localDim**self.locality,)) 



class Circuit():
    """Class that implements optimisable quantum circuits consisting of local orthorgonal matrices and acting on local Hamiltonians.

    Note:
        This class should be seen as an abstract class. 
        The action of the circuit and, thus, its geometry must be specified by implementing 
        the object functions `dotMatrix` and `dotHamiltonian` in a child class.

    Attributes:
        depth (int): depth of the circuit. 

    """
    def __init__(self):
        self.coreOrths = []
        self.depth = 0

    @property
    def coreOrths(self):
        """:obj: `list` of :obj: `numpy.array`: List of the different orthogonal matrices in the circuit. """
        return(self.__coreOrths)
    @coreOrths.setter
    def coreOrths(self,orthMatrices): 
        #TODO: implement checker for orthogonality
        self.__coreOrths = orthMatrices

    def dot(self,argument, *args, **kwargs):
        """Implements the multiplication of the circuit with given matrix or a local Hamiltionan. 

        Args: 
            argument (:obj: `numpy.array` or :obj: `Hamiltonian`): Right argument of the multiplication. 
            *args: Variable length of argument list.
            **kwargs: Arbitrary keyword arguments. 


        Note: 
            *args and **kwargs are passed to the corresponding dot routines for the specific argument types.


        Raises:
            ValueError: If `argument` is not of the allowed type.
        """
        if isinstance(argument, Hamiltonian):
            return self.dotLocalHamiltonian(argument, *args, **kwargs)
        elif isinstance(argument, np.array):
            return self.dotMatrix(argument, *args, **kwargs)
        else:
            raise ValueError('Argument type is not allowed.')


    def conjugation(self,argument, *args, **kwargs):
        """Implements the conjugation of the argument by the circuit. 

        Args: 
            argument (:obj: `numpy.array` or :obj: `Hamiltonian`): Argument of the conjugation.
            *args: Variable length of argument list.
            **kwargs: Arbitrary keyword arguments. 


        Note: 
            *args and **kwargs are passed to the corresponding conjugation routines for the specific argument types.


        Raises:
            ValueError: If `argument` is not of the allowed type.
        
        """
        if isinstance(argument, Hamiltonian):
            return self.conjugationLocalHamiltonian(argument, *args, **kwargs)
        elif isinstance(argument, np.array):
            return self.conjugationMatrix(argument, *args, **kwargs)
        else: 
            raise ValueError('Input type does not match.')


    def dotMatrix(self,givenMatrix, pos=None, locality=None, fromTheLeft=True):
        """Container for implementing the multiplication of the circuit with a matrix. 
            
            `pos` and `locality` allow to efficiently implement the multiplication of the circuit with 
            the `givenMatrix` sandwiched at position `pos` in a Kronecker product with identities of total 
            number of `locality` Hilbert spaces. 

        Args:
            givenMatrix (:obj: `numpy.array`): Argument of the multiplation. 
            pos (:obj: `int`, optional): Site the matrix is acting on.
            locality (:obj: `int`, optional): Total size of the system to be evaluated.
            fromTheLeft(:obj: `bool`, optional): Set to `False` if circuit is to be multiplied from the right. 

        Returns: 
            :obj: `numpy.array`: the result of the multiplication. 
        """

        # !!! These routine has to be implemented in a child class !!!

        pass


    def conjugationMatrix(self, givenMatrix, pos=None, locality=None):
        """Implements the conjugation of a matrix with the circuit. 
            
        `pos` and `locality` allow to efficiently implement the conjugation of 
        the `givenMatrix` sandwiched at position `pos` in a Kronecker product with identities of total 
        number of `locality` Hilbert spaces. 

        Note: The default implementation uses the classes `dotMatrix` routine. In child classes 
            it might be more efficient to directly implement the conjugation routine, especially, for 
            large values of locality.

        Args:
            givenMatrix (:obj: `numpy.array`): Argument of the multiplation. 
            pos (:obj: `int`, optional): Site the matrix is acting on.
            locality (:obj: `int`, optional): Total size of the system to be evaluated.

        Returns: 
            :obj: `numpy.array`: the result of the conjugation.
        """
        resultMatrix = self.dotMatrix(givenMatrix, pos=pos, locality=locality, fromTheLeft=True)
        resultMatrix = self.dotMatrix(resultMatrix, pos=pos, locality=locality, fromTheLeft=False)

        return resultMatrix

    def conjugationLocalHamiltonian(self, hamiltonian):
        """Implements the conjugation of local Hamiltonian with the circuit. 

        Note: 
            The current implementation only works for on-site circuits!!!! 
            More generally, the locality must be adjusted accordingly. 
            This will be done in future versions of the code. 

        Args:
            hamiltonian (:obj: `Hamiltonian`): Argument of the conjugation.

        Returns:
            :obj: `Hamiltonian`: Hamiltonian with localTerm replaced by conjugated term: circuit * localTerm * circuit.T

        """
        resultHamiltonian = self.dotLocalHamiltonian(hamiltonian, fromTheLeft=True)
        resultHamiltonian = self.dotLocalHamiltonian(resultHamiltonian, fromTheLeft=False)

        if not checkHermitian(resultHamiltonian.localTerm):
            warnings.warn("Found a non-hermitian Hamiltonian after conjugation! There is something wrong here.")


        return resultHamiltonian



    def dotLocalHamiltonian(self, hamiltonian, fromTheLeft=True):
        """Container for implementing the multiplication of the circuit with a local Hamiltonian. 
            
        Args:
            hamiltonian (:obj: `Hamiltonian`): Argument of the multiplation. 
            fromTheLeft(:obj: `bool`, optional): Set to `False` if circuit is to be multiplied from the right. 

        Returns: 
            :obj: `Hamiltonian`: the result of the multiplication. 
        """

        # !!! These routine has to be implemented in a child class !!!
        pass

    def contractWithGradient(self, gMatrix, k=0):
        """Contianer for implementing the contraction of a matrix `gMatrix`with the gradient tensor of the circuit w.r.t. the kth core orthogonal matrix.

        Typically, `gMatrix` will be the matrix representation of the outer gradient of a composed function including a circuit conjugation.

        Args:
            gMatrix (:obj: `numpy.array`): argument of the contraction. 
            k (int): the index of the core orthorgonal matrix w.r.t. that the gradient is calculated. 

        Returns: 
            :obj: `numpyp.array((circuitDim,circuitDim,orthDim,orthDim))`: Result of the contraction. 

        """

        # !!! These routine has to be implemented in a child class !!!
        pass


class Optimizer: 
    """Class that performs the optimisation of a measure for a circuit class for a given local Hamiltonian model.


    Attributes: 
        measure (:obj: `Measure`): the measure used in the optimisation. 
        hamiltonian (:obj: `Hamiltonian`): The hamiltonian model in the original basis reperesentation.
        initCircuit (:obj: `Circuit`): The circuit class and initial configurations for the core orthogonals.
    """

    def __init__(self,measure, hamiltonian, circuit=None):
        """Init method for Optimizer class. 

        Note:
            The initial circuit can also be specified later but needs to be provided before running the optimisation. 

        Args: 
            measure (:obj: `Measure`): the measure used in the optimisation. 
            hamiltonian (:obj: `Hamiltonian`): The hamiltonian model in the original basis reperesentation.
            circuit (:obj: `Circuit`, optional): The initial circuit class and initial configurations for the core orthogonals.
        """
        self.measure = measure
        self.hamiltonian = hamiltonian
        self.initCircuit = circuit


    def objectiveFunction(self,circuit):
        """Calculates the objective value of the measure at a given circuit.

        Args:
            circuit (:obj: `Circuit`): Argument for objective function. 

        Returns:
            float: Value of the objective function. 

        """
        return self.measure.evaluate(circuit.conjugationLocalHamiltonian(self.hamiltonian))


    def contractWithAdjunctionGradient(self, measureGradient, circuit):
        """Returns the gradient tensor of the conjugation of the circuit w.r.t. circuit matrix

        The argument will be typically the outer gradient matrix of the measure. 

        Args:
            measureGradient (:obj: `numpy.array`): Matrix that is contracted with the circuit's gradient tensor. 
            circuit (:obj: `Circuit`): The Circuit 

        Returns:
            :obj: `numpy.array`: The matrix representation of the result of the contraction

        """

        # retun (o \otimes o) @ h
        leftMultipliedHamiltonian = circuit.dot(self.hamiltonian) 

        # 
        firstTerm = measureGradient.dot(leftMultipliedHamiltonian.localTerm)
        secondTerm = measureGradient.transpose().dot(leftMultipliedHamiltonian.localTerm)
        contractedAdjunctionGradientMatrix =  firstTerm + secondTerm
        return(contractedAdjunctionGradientMatrix)


    def calculateEuclideanGradient(self, circuit,k):
        """Evaluates the Euclidean gradient of the objective function w.r.t. the k-th core orthogonal of the circuit. 

        Uses measure.gradient, hamiltonian.gradient and circuit.gradient to evaluate the full Euclidean gradient.
        
        Args:
            circuit (:obj: `Circuit`): Current circuit for gradient evaluation. 
            k (int): index of the circuit's core orthogonal. 

        Return:
            :obj: `numpy.array`: Matrix-valued gradient 
        """
        measureGradient = self.measure.gradient(circuit.conjugationLocalHamiltonian(self.hamiltonian)) # first calcuate O \times O　・・・ \times O @ H @ C.C , then give it as an input for calculating gradient return numpy array

        contractedAdjunctionGradientMatrix = self.contractWithAdjunctionGradient(measureGradient, circuit)
        gradientMatrix = circuit.contractWithGradient(contractedAdjunctionGradientMatrix,k)


        return gradientMatrix


    def calculateRiemannianGradient(self, circuit, k,  translated=True):
        """Calculates the Riemannian gradient of the objective function w.r.t to the k-th orthogonal matrix of circuit.

            The Riemannian gradient can be calculated from the Euclidean gradient Gamma at Q as:
                grad f(Q) = Gamma - Q*Gamma^T*Q
            It can be either evaluated in the tangent space at the actual orthogonal, the point on the manifold, or 
            translated to the tangent space of the group identity. 

            Args: 
                circuit (:obj: `numpy.array`): Current circuit 
                k (int): index of the circuit's core orthogonal. 
                translated (:obj: `bool`, optional): If True gradient is evaluated at the group identity.

        """
        euclideanGradient = self.calculateEuclideanGradient(circuit, k)
        orth = circuit.coreOrths[k]
        if translated: 
            #Gradient translated to the group identity 
            riemannianGradient = euclideanGradient.dot(orth.T)-orth.dot(euclideanGradient.T) 
        else: 
            #Gradient in the tangent space of `orth`
            riemannianGradient = euclideanGradient - orth.dot(euclideanGradient.T).dot(orth)

        return riemannianGradient


    def conjugateGradientOptimisation(self, initCircuit, k, grad_tol = 1E-8, fct_tol = 1E6, fct_goal = 0, max_steps = 150, verbose=True, retHist = None, stepper ='grid', givenStepSize = None):
        """Conjugate gradient optimisation routine for the k-th core orthogonal. 


        Implements a conjugate gradient algorithm over the orthogonal group for the k-th core orthogonal. 


        Args:
            initCircuit (:obj: `Circuit`): 
                Initial circuit of the circuit Ansatz class. 
            k (int): 
                index of the core orthogonal in the circuit to be optimised.
            grad_tol (:obj: `float`, optional): 
                Maximal value of the squared gradient norm to define local convergences. 
            fct_tol (:obj: `float`, optional): 
                Tolerated maximum value of the objective function after convergence. 
                Otherwise optimisation is restarted with different initialisation.
            fct_goal (:obj: `float`, optional): 
                Optimisation stops if objective function becomes smaller than `fct_goal`.
            max_steps (:obj: `int`, optional): 
                Maximal number of conjugate gradient steps before stoping the optimisation.  
            verbose (:obj: `bool`, optional): 
                Flag to control the std print out of the algorithm. 
            retHist (:obj: `bool`, optional): 
                If True the convergences curves of the algorithm are added to the functions result. 
            stepper (:obj: `string`, optional): 
                Chooses different modes to determine the step width. 
            givenStepSize (:obj: `float`, optional): 
                Allows to specify a fixed step width. 
            
        Returns:
        tuple of (:obj: Circuit, :obj: `dict`): 
            :obj: Circuit: is the circuit at the end of the optimisation. 
            :obj: `dict`: Dictionary of the result
                'objvar' (:obj: `numpy.array`): Optimal orthogonal matrix
                'objval' (float): Final value of the cost function
                'steps' (int): Number of conjugate gradient steps performed for optimisation
                'conv'(bool): True if the optimization converged before `max_steps` reached, False otherwise
                'log' (:obj: `numpy.array`): Array containing in the k-th row the value of the 
                        objective value, squared Riemannian norm, step size
                    of the k-th iteration. 

        """

        
        circuit = copy.deepcopy(initCircuit)
        dim = circuit.coreOrths[k].shape[0]


        # first calculation of gradient
        riemGrad = self.calculateRiemannianGradient(circuit, k, translated=True)
        riemGradNormSqrd =np.trace(np.dot(riemGrad.T,riemGrad))
        invStepDir = riemGrad

        #check Fro-norm of riemGrad > grad_tol
        smallerGradTolerance = 0.5*riemGradNormSqrd < grad_tol

        if retHist: #Initialise log object
            logHist = np.zeros((max_steps,3))

        if verbose: #Print out if wanted: 
            print("\n-----")
            print("Started conjugate gradient optimisation:")
            print("-----")
            print("obj val | grad norm | step size")

        kk = 0
        stepSize = 1E8
        while not smallerGradTolerance and kk < max_steps and not (self.objectiveFunction(circuit) < fct_goal and stepSize < 1E-7):
            kk += 1
            #Calculate step size
            if stepper == 'constant':
                stepSize = givenStepSize
            elif stepper == 'grid':
                stepSize = self.gridGetStepSize(circuit, k, invStepDir, 6)
            else:
                stepSize = self.conjugateGradientOptimisationGetStepSize(circuit, k, invStepDir, 6)
            

            
            #Go gradient step to next iterate
            circuit.coreOrths[k] = np.dot(sla.expm(-stepSize*invStepDir), circuit.coreOrths[k])

            # Hack if you get stuck: If stepSize vanishes initialise random 
            if stepSize == 0: 
                newOrth, _ = np.linalg.qr(np.random.randn(*circuit.coreOrths[k].shape)) #NOT Haar random
                circuit.coreOrths[k] = newOrth
                riemGradNormSqrd = 1E8

            
            #Calculate gradients and check tolerance 
            oldRiemGrad = riemGrad.copy()
            oldRiemGradNormSqrd = riemGradNormSqrd
            riemGrad = self.calculateRiemannianGradient(circuit, k, translated=True)
            riemGradNormSqrd = np.trace(np.dot(riemGrad.T,riemGrad))
            smallerGradTolerance = .5*riemGradNormSqrd < grad_tol

            #Approximate curvature ratio
            curvRatio = np.trace(np.dot((riemGrad-oldRiemGrad).T,riemGrad)) / oldRiemGradNormSqrd 
            
            if kk % dim*dim == 0:
                invStepDir = riemGrad
            else:
                invStepDir = riemGrad+curvRatio*invStepDir


            #check whether converged to the wrong solution:
            if smallerGradTolerance:	# Smaller than gradient tolerance
                if self.objectiveFunction(circuit) > fct_tol: #Not sufficiently converged
                    if verbose:
                        print("Trying another initialisation...")
                    smallerGradTolerance = False
                    newOrth, _ = np.linalg.qr(np.random.randn(*circuit.coreOrths[k].shape)) #NOT Haar random
                    circuit.coreOrths[k] = newOrth
                    riemGradNormSqrd = 1E8

            if verbose or retHist: 
                objVal = self.objectiveFunction(circuit)
                if verbose:
                    print(objVal, 0.5*riemGradNormSqrd, stepSize)
                if retHist:
                    logHist[kk-1] = np.array([objVal, 0.5*riemGradNormSqrd, stepSize])
                


        #Assemble results
        report = {}
        report['objvar'] = [circuit.coreOrths[k]]
        report['objval'] = self.objectiveFunction(circuit) 
        report['steps'] = kk
        if retHist:
                report['log'] = logHist
            
        if kk == max_steps and not smallerGradTolerance: #Not converged
            report['conv'] = False
            if verbose:
                print("-----\nNOT converged! NOT converged! NOT converged!\n#####\n\n")
        else:
            report['conv'] = True
            if verbose:
                print("-----\nCONVERGED after "+str(kk)+"iterations\n#####\n\n")
        return (circuit, report)


    def conjugateGradientOptimisationGetStepSize(self, circuit, k, stepDir,fct_order,polynom_order=4,showPlot =False):
        """Function for approximating the optimal stepsize during conjugated gradient with polynomial fitting.
        

        Args: 
            circuit (:obj: `Circuit`): 
                Current circuit of the circuit Ansatz class. 
            k (int): 
                index of the core orthogonal in the circuit to be optimised.
            stepDir (:obj: `numpy.array`): 
                Conjugate gradient direction.
            fct_order (int):
                Order of the cost function in the unitary (e.g if it is a 2-nd order polynomial etc)
            polynom_order (int, optional): 
                Order of the approximating polynomial which is used to estimate the stepsize.
            showPlot (bool, optional):
                If set to True a plot is produced that shows the objective function in direction of the gradient. 

        Returns
            float: Steps-size
        
        """

        orth = circuit.coreOrths[k]
        newCircuit = copy.deepcopy(circuit)

        P = polynom_order
        # get period and steps
        o  =   np.max(np.abs(np.linalg.eigvalsh(stepDir)))
        t = 2*np.pi/fct_order/o
        mu = np.arange(P+1)*t/P
        # get translations
        r_s = sla.expm(-mu[1]*stepDir)
        R = [np.identity(orth.shape[0], dtype=np.complex128)]
        for i in range(1,P+1):
            R.append(np.dot(R[-1],r_s))
        # get derivatives
        def _derivative(R):
            newCircuit.coreOrths[k] = R.dot(orth)
            return -2*np.real(np.trace(self.calculateEuclideanGradient(newCircuit, k).dot(orth.T.conj()).dot(R.T.conj()).dot(stepDir.T.conj())))
        dervs = list(map(_derivative,R))
        mu_mat_inv = np.linalg.inv(mu[1:,np.newaxis]**np.arange(1,P+1))
        coef = mu_mat_inv.dot(dervs[1:]-dervs[0])
        coef = list(coef[::-1])
        coef.append(dervs[0])
        roots = np.roots(coef)
        pos_real_roots = [np.real(r) for r in roots if np.abs(np.imag(r))<1E-10 and np.real(r)>=0]

        if showPlot is True: 
            gridSize = 200
            normPortion = .5
            orth0 = circuit.coreOrths[k]
            newCircuit = copy.deepcopy(circuit)
            gradient = stepDir

            riemGradNormSqrd = .5*np.trace(np.dot(gradient.T,gradient))
            stepSize = normPortion/ riemGradNormSqrd

            xgrid = np.arange(0,gridSize)*stepSize
            objval = np.zeros((gridSize))

            for kk in range(0,len(xgrid)):
                theOrth = np.dot(sla.expm(-xgrid[kk]*gradient), orth0)
                newCircuit.coreOrths[0] = theOrth
                objval[kk] = self.objectiveFunction(newCircuit)

            plt.plot(xgrid,objval)
            plt.show()

        if len(pos_real_roots) > 0:
            return min(pos_real_roots)
        else:		
            return 0


    def gridGetStepSize(self, circuit, k, stepDir, fct_order, gridSize = 20):
        """Determines the stepsize during conjugated gradient
        optimization using a grid search.
        
        Args: 
            circuit (:obj: `Circuit`): 
                Current circuit of the circuit Ansatz class. 
            k (int): 
                index of the core orthogonal in the circuit to be optimised.
            stepDir (:obj: `numpy.array`): 
                Conjugate gradient direction.
            fct_order (int):
                only included for compatibility with previous implementations.
            gridSize (int): size of the search grid.

        Returns
            float: Steps-size
        
        """
        
        


        orth0 = circuit.coreOrths[k]
        newCircuit = copy.deepcopy(circuit)
        gradient = stepDir
        
        _,svalues,_ = np.linalg.svd(stepDir)
        o = np.max(svalues)
        t = 10*2 * np.pi/fct_order/o

        argmin = 0
        ll = 0  
        while argmin < 3: 
            ll +=1
            t = t/10
            stepSize = t/gridSize

            xgrid = np.arange(0,gridSize)*stepSize
            objval = np.zeros((gridSize))

            if t < 1E-10:
                #print('Here I got stuck!')
                argmin = 0
                break
    
            for kk in range(0,len(xgrid)):
                theOrth = np.dot(sla.expm(-xgrid[kk]*gradient), orth0)
                newCircuit.coreOrths[0] = theOrth
                objval[kk] = self.objectiveFunction(newCircuit)
            argmin = np.argmin(objval)

        return xgrid[argmin]



    def singleOrthOptimize(self, initCircuit, k, *args, **kwargs):
        """Optimization routine for a single core orthogonal matrix.


        Wrapper to allow for more optimization routines in future. 

        Args: 
            initCircuit (:obj: `Circuit`): initial circuit for optimization.
            k (int): index of the core orthogonal matrix to be optimized. 
            *args: Variable length argument list to be passed to the optimization routine.
            **kwargs: Arbitrary keyword arguments to be passed to the optimization routine.

        Returns: 
            tuple of (:obj: Circuit, :obj: `dict`): the optimal circuit and the report of the optimization routine. 
        """
        return self.conjugateGradientOptimisation(initCircuit, k, *args, **kwargs)

    def optimize(self, *args, **kwargs):
        """Global optimizing routine over all core orthogonal matrices. 

        Args: 
            *args: Variable length argument list to be passed to the optimization routine.
            **kwargs: Arbitrary keyword arguments to be passed to the optimization routine.

        Returns: 
            tuple of (:obj: Circuit, list of :obj: `dict`): the optimal circuit and the list of reports of the optimization routine.

        Raises: 
            ValueError: If the objects `initCircuit` attribute is not specified.

        """
        

        reports = []

        if self.initCircuit is None:
            raise ValueError("No init circuit passed to optimiser. Start e.g. with random initialisation.")
        else:
            circuit = copy.deepcopy(self.initCircuit)

        # loop over all core orths and optimise
        for kk in range(0,len(circuit.coreOrths)):
            circuit, newReport = self.singleOrthOptimize(circuit, kk, *args, **kwargs)
            reports.append(newReport)

        return (circuit, reports)





# """
# Module level functions
# """

def checkUnitary(gMatrix, norm='fro', eps= 1E-8, boolean=True):
    """Returns the deviation of a matrix from being unitary.

    Args:
        gMatrix (:obj: `numpy.array`): matrix
        norm (string, optional): norm for the deviation (default Frobenius norm)
        eps (float, optional): maximal value of the norm to be considered zero if boolean is returned
        boolean (bool, optional): If True a boolean is returned, if False the norm is returned. 

    Returns: 
        if boolean=True 
            True if gUnitary is unitary, False otherwise.
        else 
            returns the the deviation in norm
    """
    deviation = np.linalg.norm(gMatrix.dot(gMatrix.conj().T) - np.eye(gMatrix.shape[0]),norm)
    if boolean:
        return(deviation <= eps)
    else:
        return(deviation)

def checkHermitian(gMatrix, norm='fro', eps= 1E-8, boolean=True):
    """Returns the deviation of a matrix from being hermitian.

    Args:
        gMatrix (:obj: `numpy.array`): matrix
        norm (string, optional): norm for the deviation (default Frobenius norm)
        eps (float, optional): maximal value of the norm to be considered zero if boolean is returned
        boolean (bool, optional): If True a boolean is returned, if False the norm is returned. 

    Returns: 
        if boolean=True 
            True if gMatrix is hermitian, False otherwise. 
        else 
            returns the the deviation in norm
    """
    deviation = np.linalg.norm(gMatrix.conj().T - gMatrix, norm)
    if boolean:
        return(deviation <= eps)
    else:
        return(deviation)


def translateMatrix(i,s,k,matrix):
    """sandwiches a matrix by identies in a Kronecker product. 


    Args:
        matrix (:obj: `numpy.array`): The matrix.
        i (int): The position of the matrix in Kronecker product.
        s (int): The total number of sites.
        k (int): The locality of the matrix.

    Returns: 
        :obj: `numpy.array`: matrix representation of the resulting Kronecker product. 
    """
    localDim = int(round(np.exp(np.log(matrix.shape[0])/k)))

    reshapeIndices = 2 * s * (localDim,)
    return np.kron(np.kron(np.eye(localDim**i),matrix),np.eye(localDim**(s - i - k))).reshape(reshapeIndices)
