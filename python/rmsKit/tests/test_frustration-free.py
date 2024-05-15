import torch
import numpy as np
import pytest
import logging
from typing import List

from ..lattice import FF
from ..utils import sum_ham


logger = logging.getLogger(__name__)



class TestFF1D:
    def setup_method(self):
        self.params1 = {
            "sps": 3,
            "rank": 2,
            "dimension": 1,
            "seed": 42,
            "lt": 1
        }

        self.params2 = {
            "sps": 4,
            "rank": 3,
            "dimension": 1,
            "seed": 42,
            "lt": 1
        }

        self.params3 = {
            "sps": 3,
            "rank": 4,
            "dimension": 1,
            "seed": 42,
            "lt": 2
        }

    def test_shape(self):
        h, sps = FF.local(self.params1)
        assert h.shape == (1, self.params1["sps"]**(2*self.params1["lt"]), self.params1["sps"]**(2*self.params1["lt"])), "Unexpected shape of returned matrix"
        assert sps == self.params1["sps"]**self.params1["lt"], "Unexpected sps value"
        assert np.allclose(h, h.transpose(0, 2, 1).conj()), "Returned matrix is not symmetric"

        logger.debug("Shape : {}".format(h.shape))
        # Check ground state energy is 0
        E, V = np.linalg.eigh(h[0])
        assert np.allclose(E[0], 0), "Ground state energy is not null"

        # Check energy spectrum is interger valued
        assert np.allclose(E, np.round(E)), "Energy spectrum is not integer valued"



        h, sps = FF.local(self.params2)
        assert h.shape == (1, self.params2["sps"]**(2*self.params2["lt"]), self.params2["sps"]**(2*self.params2["lt"])), "Unexpected shape of returned matrix"
        assert sps == self.params2["sps"]**self.params2["lt"], "Unexpected sps value"
        assert np.allclose(h, h.transpose(0, 2, 1).conj()), "Returned matrix is not symmetric"
        #Check h is not null matrix 
        assert not np.allclose(h, np.zeros_like(h)), "Returned matrix is null"

        # Check ground state energy is 0
        E, V = np.linalg.eigh(h[0])
        assert np.allclose(E[0], 0), "Ground state energy is not null"

        # Check energy spectrum is interger valued
        assert np.allclose(E, np.round(E)), "Energy spectrum is not integer valued"


        #hamiltonian should be null
        h, sps = FF.local(self.params3)
        assert h.shape == (1, self.params3["sps"]**(2*self.params3["lt"]), self.params3["sps"]**(2*self.params3["lt"])), "Unexpected shape of returned matrix"
        assert sps == self.params3["sps"]**self.params3["lt"], "Unexpected sps value"
        assert np.allclose(h, np.zeros_like(h)), "Returned matrix is not null"

    def test_ff_property(self):
        h, sps = FF.local(self.params1)
        L = 7
        H = sum_ham(h[0], [[i, i+1] for i in range(L-1)], L, sps)
        assert np.allclose(H, H.T.conj())
        E, V = np.linalg.eigh(H)
        assert np.allclose(E[0], 0), "Ground state energy is not null"

        h, sps = FF.local(self.params2)
        L = 5
        H = sum_ham(h[0], [[i, i+1] for i in range(L-1)], L, sps)
        assert np.allclose(H, H.T.conj())
        E, V = np.linalg.eigh(H)
        assert np.allclose(E[0], 0), "Ground state energy is not null"




