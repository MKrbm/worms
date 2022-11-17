import numpy as np
from scipy import sparse
import os
import sys
sys.path.insert(0, "../") 
sys.path.insert(0, "../../nsp") 
from nsp.utils.func import *
from nsp.utils.print import beauty_array
from save_npy import *
import argparse

lattice = [
    "original",
]

loss = ["mes", "l1"]

parser = argparse.ArgumentParser(description='Optimize majumdar gosh')
parser.add_argument('-l','--lattice', help='lattice (model) Name', required=True, choices=lattice)
args = vars(parser.parse_args())
lat = args["lattice"]

Sz = np.zeros([2,2])
Sz[0,0] = 1/2
Sz[1,1] = -1/2
Sx = np.zeros([2,2])
Sx[1,0] = 1/2
Sx[0,1] = 1/2
Sy = np.zeros([2,2], dtype=np.complex64)
Sy[1,0] = 1j/2
Sy[0,1] = -1j/2


SzSz = np.kron(Sz,Sz).real.astype(np.float64)
SxSx = np.kron(Sx,Sx).real.astype(np.float64)
SySy = np.kron(Sy,Sy).real.astype(np.float64)

lh = SzSz + SxSx + SySy

lh = -lh # use minus of local hamiltonian for monte-carlo (exp(-beta H ))

if lat == "original":
    H = lh
    path = ["../../array/J1J2/original"]
    save_npy(path[0], [H, H])





