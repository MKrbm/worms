import torch
import numpy as np
import pytest
import logging
from typing import List

from ..lattice import FF, get_model
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
            "rank": 2,
            "dimension": 1,
            "seed": 42,
            "lt": 1
        }

        self.params4 = {
            "sps": 9,
            "rank": 2,
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
    
        L = 3
        h, sps = FF.local(self.params4)
        assert not np.allclose(h, np.zeros_like(h)), "Returned matrix is null"
        H = sum_ham(h[0], [[i, i+1] for i in range(L-1)], L, sps)
        assert np.allclose(H, H.T.conj())
        E, V = np.linalg.eigh(H)
        assert np.allclose(E[0], 0), "Ground state energy is not null"

    def test_ff_system(self):
        L = 6
        H, sps = FF.system([L], params=self.params1)
        assert H.shape == (sps**L, sps**L), "Unexpected shape of returned matrix"
        assert sps == self.params1["sps"]**self.params1["lt"], "Unexpected sps value"
        assert np.allclose(H, H.T.conj()), "Returned matrix is not symmetric"
        E, V = np.linalg.eigh(H)
        assert np.allclose(E[0], 0), "Ground state energy is not null"

        # test sum_ham return the same hamiltonian
        h, sps = FF.local(self.params1)
        H_ = sum_ham(h[0], [[i, (i+1)%L ] for i in range(L)], L, sps)
        assert np.allclose(H, H_), "Returned matrix is not the same"

        #print all ground state energy
        # print(E)
    
    def test_get_model(self):
        params = {
            "sps": 3,
            "rank": 2, #get_model always use rank : 2
            "dimension": 1,
            "seed": 42,
            "lt": 1
        }
        get_model_params = {
            "sps" : params["sps"],
            "seed" : params["seed"],
            "lt" : params["lt"],
        }
        h1, sps1 = FF.local(params)
        h2, sps2, model_name  = get_model("FF1D", params=get_model_params)

        assert np.allclose(h1, h2), "Returned matrix is not the same"
        assert sps1 == sps2, "Unexpected sps value"

        assert model_name == f"FF1D_loc/s_{params['sps']}_r_{params['rank']}_d_{params['dimension']}_seed_{params['seed']}", "Unexpected model name"

        logger.debug(model_name)
        



