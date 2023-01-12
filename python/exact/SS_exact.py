import sys,os
import numpy as np
from scipy import sparse

from quspin.basis import spin_basis_1d, spin_basis_general
from quspin.operators import hamiltonian,quantum_operator
from quspin.tools.lanczos import lanczos_full,lanczos_iter,FTLM_static_iteration,LTLM_static_iteration
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian,quantum_operator
from quspin.tools.lanczos import lanczos_full,lanczos_iter,FTLM_static_iteration,LTLM_static_iteration
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

parser = argparse.ArgumentParser(description='exact diagonalization of shastry_surtherland')
parser.add_argument('-J','--coupling', help='coupling constant (NN)', type = float, default = 1)
args = parser.parse_args()

'''
calculate exact 4 x 4 shastry model
'''

class lanczos_wrapper(object):
    """
    Class that contains minimum requirments to use Lanczos. 
    
    Using it is equired, since the dot and dtype methods of quantum_operator objects take more parameters 
    
    """
    #
    def __init__(self,A,**kwargs):
        """
        A: array-like object to assign/overwrite the dot and dtype objects of
        kwargs: any optional arguments used when overwriting the methods

        """
        self._A = A
        self._kwargs = kwargs
    #
    def dot(self,v,out=None):
        """
        Calls the `dot` method of quantum_operator with the parameters fixed to a given value.

        """
        return self._A.dot(v,out=out,pars=self._kwargs)
    #
    @property
    def dtype(self):
        """
        The dtype attribute is required to figure out result types in lanczos calculations.

        """
        return self._A.dtype


Lx, Ly = 4, 4 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
Z   = -(s+1) # spin inversion
# basis_2d = spin_basis_general(N_2d,pauli=False)
spin_basis_general(N_2d,kxblock=(T_x,0),kyblock=(T_y,0),pxblock=(P_x,0),pyblock=(P_y,0),zblock=(Z,0), pauli=True)

# set up coupling 
J1 = args.coupling
J2 = 1
J_zz = [[J1,i,T_x[i]] for i in range(N_2d)]+[[J1,i,T_y[i]] for i in range(N_2d)] 
J_zz = J_zz + [[J2,0,5],[J2,2,7],[J2,8,13],[J2,10,15],[J2,1,14],[J2,3,12],[J2,6,9],[J2,4,11]]
J_xy = [[J1/2.0,i,T_x[i]] for i in range(N_2d)]+[[J1/2.0,i,T_y[i]] for i in range(N_2d)]
J_xy = J_xy + [[J2/2.0,0,5],[J2/2.0,2,7],[J2/2.0,8,13],[J2/2.0,10,15],[J2/2.0,1,14],[J2/2.0,3,12],[J2/2.0,6,9],[J2/2.0,4,11]]
ops_dict = dict(Jpm=[["+-",J_xy]],Jmp=[["-+",J_xy]],Jzz=[["zz",J_zz]])

E = []
# for i in range(Lx*Ly+1):
basis_2d = spin_basis_general(N_2d ,pauli=False)
H = quantum_operator(ops_dict,basis=basis_2d,dtype=np.float64, check_symm=False)

L = N_2d # system size
N_samples = 50 # of samples to approximate thermal expectation value with
m = 50 # dimensio of Krylov space
T = np.logspace(-1.6, 0.3, num=50) # temperature vector
beta = 1.0/(T+1e-15) # inverse temperature vector
H_wrapped = lanczos_wrapper(H)
out = np.zeros((m,H.Ns),dtype=np.float64)
[E0] = H.eigsh(k=1,which="SA",return_eigenvectors=False)
E_list = []
Vs = []
lvs = []
for i in range(N_samples):
    r = np.random.normal(0,1,size=H.Ns)
    r /= np.linalg.norm(r)
    E,V,lv = lanczos_full(H,r,m,eps=1e-8,full_ortho=True)
    E -= E0
    E_list.append(E)
    Vs.append(V)
    lvs.append(lv)

np.save(f"npy/SS_E_J_{J1}_N_{m}x{N_samples}", E_list)
np.save(f"npy/SS_V_J_{J1}_N_{m}x{N_samples}", Vs)
np.save(f"npy/SS_lv_J_{J1}_N_{m}x{N_samples}", lvs)

import pickle
with open('npy/SS_H_J_{J1}.pickle', 'wb') as handle:
    pickle.dump(H, handle, protocol=pickle.HIGHEST_PROTOCOL)
