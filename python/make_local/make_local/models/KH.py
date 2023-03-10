# kagome heisenberg model

import sys,os
import numpy as np
# from ..core import *
from ..core.constants import *
from ..core.utils import *
sys.path.append('../reduce_nsp')
from nsp.utils import l2nl, sum_ham, cast_local_jax
from nsp.utils.func import *
from nsp.utils.local2global import *
from nsp.utils.print import beauty_array

unitary_algorithms = ["original", "3site", "3site_diag"]

# return minus of local hamiltonian
def KH(ua : str, params : dict):
    if ua not in unitary_algorithms:
        raise ValueError("unitary_algorithms not supported")
    Jz = params["Jz"]
    Jx = params["Jx"]
    Jy = params["Jy"]
    hz = params["hz"]
    hx = params["hx"]
    h_bond = Jz*SzSz + Jx*SxSx + Jy*SySy 
    h_single = hz*Sz + hx*Sx

    if ua == "original":
        h = h_bond + (np.kron(h_single, I2) +  np.kron(I2, h_single)) / 4.0 #n* there is 4 bond per site
        sps = 2
        return [-h], sps

    if ua == "3site":
        _h = sum_ham(h_bond / 6, [
            [0,1],[1,2],[2,0], [3,4], [4,5], [5,3]
        ], 6, 2) 
        _h += sum_ham(h_single / 6, [[i] for i in range(6)], 6, 2) #n* there is 6 bond per site
        
        h = [
            _h + sum_ham(h_bond, [[0, 5]], 6, 2),
            _h + sum_ham(h_bond, [[0, 4]], 6, 2),
            _h + sum_ham(h_bond, [[2, 4]], 6, 2),
        ]
        return [-_h for _h in h], 8
        # rewrite this with forloop 

    if ua == "3site_diag":
        _h0 = sum_ham(h_bond , [
            [0,1],[1,2],[2,0]
        ], 3, 2) 
        E, V = np.linalg.eigh(_h0)
        U = np.kron(V, V)
        _h = sum_ham(h_bond / 6, [
            [0,1],[1,2],[2,0], [3,4], [4,5], [5,3]
        ], 6, 2) 
        _h += sum_ham(h_single / 6, [[i] for i in range(6)], 6, 2) #n* there is 6 bond per site
        
        # h = [
        #     _h + sum_ham(h_bond, [[0, 5]], 6, 2),
        #     _h + sum_ham(h_bond, [[0, 4]], 6, 2),
        #     _h + sum_ham(h_bond, [[2, 4]], 6, 2),
        # ]
        h = [
            _h, 
            _h, 
            _h, 
        ]
        # h = [-U.T @ _h @ U for _h in h]

        #n* symmetrize
        h = [(_h + _h.T) / 2 for _h in h]
        return [-_h for _h in h], 8
        # rewrite this with forloop 

    if ua == "6site":
        print("impossible")
        # _h = sum_ham(h_bond, [
        #     [0,1],[1,2],[2,0], [3,4], [4,5], [5,3]
        # ], 6, 2) 
        # -h2 = sum_ham(_h, [[0], [1]], 2, 8)

        # h = h_bond + (np.kron(h_single, I2) +  np.kron(I2, h_single)) / 4.0
    
    print("Other unitary_algorithms mode not implemented")
    return None, None
        # if loss is not "none":
        #     raise ValueError("loss not supported")

        


