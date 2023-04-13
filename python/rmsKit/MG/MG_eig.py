import numpy as np
import argparse
import sys
sys.path.append('../nsp')
from nsp.utils.func import *

parser = argparse.ArgumentParser(description='Reproduce original paper results')
parser.add_argument('-t','--T', help='temperature', required=False, type=float, default = 1)
parser.add_argument('-l','--L', help='system size', required=False, type=int, default = 6)
parser.add_argument('-c','--C', help='1D chain lattice or not', action='store_true', default = False)

args = vars(parser.parse_args())
t = args["T"]
L = args["L"]
chain = args["C"]

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

if chain:
    print("1D chain lattice")
    assert L%3 == 0, "L should be 3*K "
    bonds = [[0,1], [0, 2], [1, 2]]
    lh2 = sum_ham(lh, bonds, 3, 2)
    LH = sum_ham(lh2/2, [[0,1,2], [3, 4, 5]], 6, 2) + sum_ham(lh2, [[1, 2, 3], [2, 3, 4]], 6, 2)
    bonds = [[i, (i+1)%int(L/3)] for i in range(int(L/3))]
    H = sum_ham(LH, [[0, 1], [1, 0]], 2, 8)
    if L/3 == 2:
        H = H/2
else:
    print("original MG")
    bonds1 = [[i, (i+1)%L] for i in range(L)]
    bonds2 = [[i, (i+2)%L] for i in range(L)]
    H = sum_ham(lh, bonds1, L, 2) + sum_ham(lh/2, bonds2, L, 2)
E = np.linalg.eigvalsh(H)
Z = np.exp(-1/t * E)

print(f"at T = {t}")
print(f"expected energy : {(Z*E).sum()/Z.sum()}")

