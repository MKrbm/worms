import sys,os
import numpy as np

from quspin.basis import spin_basis_1d, spin_basis_general
from quspin.operators import hamiltonian,quantum_operator
from quspin.tools.lanczos import lanczos_full,lanczos_iter,FTLM_static_iteration,LTLM_static_iteration

import argparse
from core import lancoz_estimate


os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='12' # set number of MKL threads to run in parallel

lattice = [
    "1D",
    "2D"
]

parser = argparse.ArgumentParser(description='exact diagonalization of shastry_surtherland')
parser.add_argument('-l','--lattice', help='lattice (model) Name', required=True, choices=lattice)
parser.add_argument('-Jz','--coupling_z', help='coupling constant (NN)', type = float, default = 1)
parser.add_argument('-Jx','--coupling_x', help='coupling constant (NN)', type = float, default = 1)

parser.add_argument('-H','--magnetic', help='magnetic field', type = float, default = 0)
parser.add_argument('-n','--n_samples', help='# of samples', type = int, default = 50)
parser.add_argument('-m','--n_Krylov', help='dimernsion of Krylov space', type = int, default = 50)
parser.add_argument('-T', "--temperature", help = "temperature", type = float)
parser.add_argument('-L', "--length", help = "length of side", type = int, default = 4)


args = parser.parse_args()
lat = args.lattice
T = args.temperature
L = args.lenght
if T is None:
    T = np.concatenate([np.logspace(-1.6, 0, num=80)[:-1], np.logspace(0, 2, num = 20)])
    printout = False
else:
    if T < 0:
        raise ValueError("Temperature must be positive")
    T = np.array([T])
    printout = True

# H = J ( \sum_{<ij>} \sigma^z_i \sigma^z_j + \delta (<x> + <y>) )- h \sum_i \sigma^z_i


if lat == "2D":
    Lx, Ly = L, L
    N_2d = Lx*Ly 
    if N_2d >= 16:
        print("Warning: N_2d is too large for exact diagonalization. Please use Lanczos method instead")
    s = np.arange(N_2d)
    x = s%Lx 
    y = s//Lx 
    T_x = (x+1)%Lx + Lx*y 
    T_y = x +Lx*((y+1)%Ly) 

    Jz = args.coupling_z
    Jx = args.coupling_x
    h = args.magnetic
    N = args.n_samples
    K = args.n_Krylov


    J_zz = [[1,i,T_x[i]] for i in range(N_2d)]+[[1,i,T_y[i]] for i in range(N_2d)] 
    J_xy = [[1,i,T_x[i]] for i in range(N_2d)]+[[1,i,T_y[i]] for i in range(N_2d)]
    h_list = [[-1.0,i] for i in range(N_2d)]
    ops_dict = dict(Jz=[["zz", J_zz]], # J for z
                    Jx=[["yy",J_xy], ["xx",J_xy]], # J for x and y
                    h=[["z",h_list]])
    M_list = [[1.0/N_2d,i] for i in range(N_2d)]

    basis_2d = spin_basis_general(N_2d ,pauli=False)
    H_SS = quantum_operator(ops_dict,basis=basis_2d, dtype=np.float64, check_symm=False)
    M = hamiltonian([["z",M_list]],[],basis=basis_2d, dtype=np.float64)

    # define magnetization observable


    # define hamiltonian for given parameters
    params_dict=dict(Jz=Jz,Jx=Jx,h=h)

    # print(T)
    beta = 1.0/(T+1e-15) 

    lancoz_estimate(H_SS, params_dict, M, N, K, beta, model_name="HXXZ2D", printout  = printout)

elif lat == "1D":

    L = 10

    Jz = args.coupling_z
    Jx = args.coupling_x
    h = args.magnetic
    N = args.n_samples
    K = args.n_Krylov


    J_zz = [[1,i,(i+1)%L] for i in range(L)]
    J_xy = [[1,i,(i+1)%L] for i in range(L)]
    h_list = [[-1.0,i] for i in range(L)]
    ops_dict = dict(Jz=[["zz", J_zz]], # J for z
                    Jx=[["yy",J_xy], ["xx",J_xy]], # J for x and y
                    h=[["z",h_list]])
    M_list = [[1.0/L,i] for i in range(L)]

    basis_2d = spin_basis_1d(L ,pauli=False)
    H_SS = quantum_operator(ops_dict,basis=basis_2d, dtype=np.float64, check_symm=False)
    M = hamiltonian([["z",M_list]],[],basis=basis_2d, dtype=np.float64)

    # define magnetization observable


    # define hamiltonian for given parameters
    params_dict=dict(Jz=Jz,Jx=Jx,h=h)

    # print(T)
    beta = 1.0/(T+1e-15) 

    lancoz_estimate(H_SS, params_dict, M, N, K, beta, model_name="HXXZ1D", printout  = printout)
