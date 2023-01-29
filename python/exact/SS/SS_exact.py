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


parser = argparse.ArgumentParser(description='exact diagonalization of shastry_surtherland')
parser.add_argument('-J','--coupling', help='coupling constant (NN)', type = float, default = 1)
parser.add_argument('-H','--magnetic', help='magnetic field', type = float, default = 0)
parser.add_argument('-n','--n_samples', help='# of samples', type = int, default = 50)
parser.add_argument('-m','--n_Krylov', help='dimernsion of Krylov space', type = int, default = 50)

args = parser.parse_args()


Lx, Ly = 4, 4 
N_2d = Lx*Ly 
s = np.arange(N_2d)
x = s%Lx 
y = s//Lx 
T_x = (x+1)%Lx + Lx*y 
T_y = x +Lx*((y+1)%Ly) 

J1 = args.coupling 
h = args.magnetic
N = args.n_samples
K = args.n_Krylov


J_zz_p = [[1,i,T_x[i]] for i in range(N_2d)]+[[1,i,T_y[i]] for i in range(N_2d)] 
J_zz_x = [[1,0,5],[1,2,7],[1,8,13],[1,10,15],[1,1,14],[1,3,12],[1,6,9],[1,4,11]]
J_xy_p = [[1/2.0,i,T_x[i]] for i in range(N_2d)]+[[1/2.0,i,T_y[i]] for i in range(N_2d)]
J_xy_x = [[1/2.0,0,5],[1/2.0,2,7],[1/2.0,8,13],[1/2.0,10,15],[1/2.0,1,14],[1/2.0,3,12],[1/2.0,6,9],[1/2.0,4,11]]
h_list = [[-1.0,i] for i in range(N_2d)]
ops_dict = dict(Jp=[["+-",J_xy_p], ["-+",J_xy_p], ["zz", J_zz_p]], # J_perpendicular 
                Jx=[["+-",J_xy_x], ["-+",J_xy_x], ["zz", J_zz_x]],
                h=[["z",h_list]])

basis_2d = spin_basis_general(N_2d ,pauli=False)
H_SS = quantum_operator(ops_dict,basis=basis_2d, dtype=np.float64, check_symm=False)

# define magnetization observable
M_list = [[1.0/N_2d,i] for i in range(N_2d)]
M = hamiltonian([["z",M_list]],[],basis=basis_2d, dtype=np.float64)
M2 = M**2


# define hamiltonian for given parameters
params_dict=dict(Jp=J1,h=h)

T = np.logspace(-1.6, 0.3, num=80) 
beta = 1.0/(T+1e-15) 

lancoz_estimate(H_SS, params_dict, M, N, K, beta, model_name="SS")

