# -*- coding: utf-8 -*-
"""identical Onsite Circuit

This module provides a subclass of `Circuit` that implements a single orthogonal matrix 
simultaneously action on each site. 

This Ansatz class was used in the numerical simulation of the publication

    Hangleiter, Roth, Nagaj, and Eisert. "Easing the Monte Carlo sign problem" 
    https://arxiv.org/abs/1906.02309

Authors: 
    Dominik Hangleiter and Ingo Roth

Licence:
    This project is licensed under the MIT License - see the LICENSE.md file for details.
"""
from circuitOptimizer import *

class IdenticalOnsiteCircuit(Circuit):
    """A quantum circuit that consist of a single orthogonal matrix repeated over all sites.
        

        The class implements a quantum circuit of the form 
            O x O x ... x O x O 
        where the same orthorgonal matrix O acts on ever local Hilbert space. 

    """
    def __init__(self, localDim=0):
        """Init routine for an identical onsite circuit.

        The core orthogonal matrix is initialized as the identity matrix. 
        """
        super().__init__()
        self.coreOrths = [np.eye(localDim)] 
        self.depth = 1	#int: circuit depth is allways one.


    @property
    def coreOrths(self):
        """:obj: `list` of :obj: `numpy.array`: List consisting of the single orthogonal matrix definining the circuit"""
        return(self.__coreOrths)
    @coreOrths.setter
    def coreOrths(self,orthMatrices): 
        self.__coreOrths = orthMatrices

    @property
    def localDim(self):
        """int: local dimension of the Hilbert spaces infered from the core orthogonal matrix."""
        return self.coreOrths[0].shape[0]
    


    def initRandom(self, localDim=None, seed = None):
        """Initialises the core orthogonal matrix with a Haar random orthogonal matrix.

        Args: 
            localDim (:obj: `int`, optional): Sets the dimension of the orthogonal matrix, 
                otherwise the object's `localDim` property is used.
            seed (:obj: `int`, optional): Seed for random generator passed to `numpy.random.seed`.

        """
        if localDim is None: 
            localDim = self.coreOrths[0].shape[0]

        np.random.seed(seed)
            
        randnMatrix = np.random.randn(localDim, localDim)

        Q, R = np.linalg.qr(randnMatrix)
    
        haarOrth = Q.dot(np.diag(np.sign(np.diag(R)))) #Correct for signs of diagonal entries of R 

        self.coreOrths[0] = haarOrth


    def dotMatrix(self, givenMatrix, pos=None, locality=None, fromTheLeft=True):
        """Multiplication of the circuit with a matrix. 
            
        Note:
            `pos` AND `locality` functionality is NOT implemented yet!

        Args:
            givenMatrix (:obj: `numpy.array`): Argument of the multiplation. 
            pos (:obj: `int`, optional): Site the matrix is acting on. (NOT implemented!)
            locality (:obj: `int`, optional): Total size of the system to be evaluated. (NOT implemented!)
            fromTheLeft(:obj: `bool`, optional): Set to `False` if circuit is to be multiplied from the right. 

        Returns: 
            :obj: `numpy.array`: the result of the multiplication. 

        Raises: 
            ValueError: If arguments `pos` or `locality` are specified. 
            ValueError: If dimension of the givenMatrix does not match the local dimension of the orthogonal matrix.

        """


        if not (pos is None and locality is None):
            raise ValueError('pos and locality not implemented for onsite-circuit')

        localDim = self.localDim
        locality = int(round(np.log(givenMatrix.shape[0])/np.log(localDim)))

        if localDim**locality != givenMatrix.shape[0]: 
            raise ValueError('Dimension mismatch between coreOrths and localDim.')

        resultMatrix = givenMatrix.copy()

        theOrth = self.coreOrths[0] if fromTheLeft else self.coreOrths[0].transpose()
        for kk in range(locality):
            resultMatrix = self.multiplyTensorLegWithMatrix(resultMatrix, theOrth, kk, fromTheLeft=fromTheLeft)
        
        return resultMatrix

    def dotLocalHamiltonian(self, hamiltonian, fromTheLeft=True):
        """Multiplication of a local Hamiltonianw with the correct circuit patch.
            
        Args:
            hamiltonian (:obj: `Hamiltonian`): Argument of the multiplation. 
            fromTheLeft(:obj: `bool`, optional): Set to `False` if circuit is to be multiplied from the right. 

        Returns: 
            :obj: `Hamiltonian`: the result of the multiplication. 
        """

        resultMatrix = self.dotMatrix(hamiltonian.localTerm, fromTheLeft=fromTheLeft)
        newHamiltonian = Hamiltonian(aMatrix=resultMatrix, localDim=hamiltonian.localDim,locality=hamiltonian.locality)
        return newHamiltonian


    def contractWithGradient(self, gMatrix, k=0):
        """Calculates the contraction of  a matrix `gMatrix`with the gradient tensor of the circuit w.r.t. the core orthogonal matrix.

        Args:
            gMatrix (:obj: `numpy.array`): argument of the contraction. 
            k (:obj: int, optional): should be always 0 for identical onsite circuit. 

        Returns: 
            :obj: `numpyp.array((circuitDim,circuitDim,orthDim,orthDim))`: Result of the contraction. 

        """


        if k >= len(self.coreOrths):
            warnings.warn("Core Corthogonal index "+k+" out of range.")
        
        theOrth = self.coreOrths[0]
        localDim = theOrth.shape[0]
        locality = int(round(np.log(gMatrix.shape[0])/np.log(localDim)))


        contractedWithCircuitGradient = np.zeros((localDim,localDim))
        for kk in range(0,locality):
            partiallyContractedTensor =  gMatrix
            for ll in range(0,locality-1): #subsequently contract all legs besides the kk-th one with the Orth
                partiallyContractedTensor = self.contractTensorLegWithMatrix(partiallyContractedTensor, theOrth.transpose(), int(ll >= kk))

            contractedWithCircuitGradient = contractedWithCircuitGradient + partiallyContractedTensor

        return contractedWithCircuitGradient


    def multiplyTensorLegWithMatrix(self, tensorMatrix, gMatrix, legIdx,fromTheLeft= True):
        """Multiplies the leg with index `legIdx of an (ddd...d)x(ddd..d) tensor with a dxd `gMatrix` from the left or right. 

        The method is used to calculate dot products with the circuit. 
        The result is matrix multiplication
            (Id x ... x Id x gMatrix x Id x ... x Id) * tensorMatrix
        with gMatrix at position `legIdx` and x the Kronecker product. 


        Args: 
            tensorMatrix (:obj: `numpy.array`): The matrix representation of the tensor. 
            gMatrix (:obj: `numpy.array`): The matrix to multiply with.
            legIdx (int): The position of the matrix multiplication.
            fromTheLeft (:obj: `bool`, optional): Set to False for multiplication with the matrix from the right.

        Returns:
            :obj: `numpy.array`: Matrix representation of the resulting tensor
        """
        localDim = gMatrix.shape[0]
        locality = int(round(np.log(tensorMatrix.shape[0])/np.log(localDim)))


        tmpTensor = tensorMatrix.reshape(((localDim**legIdx, localDim, localDim**(locality-legIdx-1))*2))
        if fromTheLeft:
            contractedTensor = np.einsum("aibcjd, ki->akbcjd", tmpTensor, gMatrix)
        else: 
            contractedTensor = np.einsum("aibcjd, jk->aibckd", tmpTensor, gMatrix)

        contractedMatrix = contractedTensor.reshape((localDim**locality, localDim**locality))

        return contractedMatrix

    def contractTensorLegWithMatrix(self, tensorMatrix, gMatrix, legIdx):
        """Contracts the leg with index `legIdx` of an (ddd...d)x(ddd..d) tensor with a dxd `gMatrix`.

        The method is used to calculate the gradient contraction with the circuit. 
        The action is similar to the `multiplyTensorLegWithMatrix` function but also takes the partial 
        trace with respect to the Hilbert space with index `legIdx` after the multiplication. 

        Args: 
            tensorMatrix (:obj: `numpy.array`): The matrix representation of the tensor. 
            gMatrix (:obj: `numpy.array`): The matrix to contract with the tensor.
            legIdx (int): The position of the contraction on the tensor.

        Returns:
            :obj: `numpy.array`: Matrix representation of the resulting tensor
        """

        localDim = gMatrix.shape[0]
        locality = int(round(np.log(tensorMatrix.shape[0])/np.log(localDim)))

        tmpTensor = tensorMatrix.reshape(((localDim**legIdx, localDim, localDim**(locality-legIdx-1))*2))

        contractedTensor = np.einsum("aibcjd, ji", tmpTensor, gMatrix)

        locality = locality - 1
        contractedMatrix = contractedTensor.reshape((localDim**locality, localDim**locality)) 

        return contractedMatrix