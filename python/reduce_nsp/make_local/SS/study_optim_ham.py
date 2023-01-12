import numpy as np
import argparse
import sys
sys.path.append('../..')
from nsp.utils.base_conv import *
from header import *
from nsp.utils.func import *
from nsp.utils.local2global import *
from nsp.utils.print import beauty_array

loss = ["mes", "l1"]
J = [1, 1] #J, J_D

parser = argparse.ArgumentParser(description='for studying SS optimized hamiltonian')
parser.add_argument('-J','--coupling', help='coupling constant (NN)', type = float, default = 1.0)
args = parser.parse_args()
print(args)
J[0] = float(args.coupling)
J_str = str(J).replace(" ", "")
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


u = np.array([
    [0,1,0,0],
    [1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [-1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [0,0,0,1]
])

U = np.kron(u, u)
print(J)
H1 = sum_ham(J[0]*lh, [[1,2],[1,3]], 4, 2)
H1 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)

print(sum_ham(lh, [[1,2],[1,3]], 4, 2))
print(sum_ham(lh/4, [[0,1],[2,3]], 4, 2))

H2 = sum_ham(J[0]*lh, [[0,2],[0,3]], 4, 2)
H2 += sum_ham(J[1]*lh/4, [[0,1],[2,3]], 4, 2)


H1_optim = np.load(f"array/dimer_optim_J_{J_str}_M_{240}/0.npy")
H2_optim = np.load(f"array/dimer_optim_J_{J_str}_M_{240}/1.npy")
H1_singlet = np.load(f"array/singlet_J_{J_str}/0.npy")
H2_singlet = np.load(f"array/singlet_J_{J_str}/1.npy")

from contextlib import contextmanager
@contextmanager
def print_array_on_one_line():
    oldoptions = np.get_printoptions()
    np.set_printoptions(linewidth=np.inf)
    yield
    np.set_printoptions(**oldoptions)

with print_array_on_one_line():
    print("energy spectrum of H1 : \n" , np.linalg.eigvalsh(-H1))
    # print("energy spectrum of stochastic H1 : \n" , np.linalg.eigvalsh(-stoquastic(H1)) - np.linalg.eigvalsh(-H1))
    # print("energy spectrum of stochastic H1_singlet : \n" , np.linalg.eigvalsh(-stoquastic(H1_singlet)) - np.linalg.eigvalsh(-H1))
    # print("energy spectrum of stochastic H1_optim : \n" , np.linalg.eigvalsh(-stoquastic(H1_optim)) - np.linalg.eigvalsh(-H1))
    print("energy spectrum of stochastic H1 : \n" , np.linalg.eigvalsh(-stoquastic(H1)) )
    print("energy spectrum of stochastic H1_singlet : \n" , np.linalg.eigvalsh(-stoquastic(H1_singlet)) )
    print("energy spectrum of stochastic H1_optim : \n" , np.linalg.eigvalsh(-stoquastic(H1_optim)) )

    print("\n")

    print("energy spectrum of H2 : \n" , np.linalg.eigvalsh(-H2))
    # print("energy spectrum of stochastic H2 : \n" , np.linalg.eigvalsh(-stoquastic(H2)) - np.linalg.eigvalsh(-H2))
    # print("energy spectrum of stochastic H2_singlet : \n" , np.linalg.eigvalsh(-stoquastic(H1_singlet)) - np.linalg.eigvalsh(-H2))
    # print("energy spectrum of stochastic H2_optim : \n" , np.linalg.eigvalsh(-stoquastic(H2_optim)) - np.linalg.eigvalsh(-H2))

    print("energy spectrum of stochastic H2 : \n" , np.linalg.eigvalsh(-stoquastic(H2)) )
    print("energy spectrum of stochastic H2_singlet : \n" , np.linalg.eigvalsh(-stoquastic(H2_singlet)) )
    print("energy spectrum of stochastic H2_optim : \n" , np.linalg.eigvalsh(-stoquastic(H2_optim)) )