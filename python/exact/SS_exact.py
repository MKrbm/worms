import numpy as np
from scipy import sparse

from quspin.basis import spin_basis_1d, spin_basis_general
from quspin.operators import hamiltonian,quantum_operator
from quspin.tools.lanczos import lanczos_full,lanczos_iter,FTLM_static_iteration,LTLM_static_iteration
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse
parser = argparse.ArgumentParser(description='exact diagonalization of shastry_surtherland')
parser.add_argument('-J','--coupling', help='coupling constant (NN)', type = float, default = 1)
args = parser.parse_args()

'''
calculate exact 4 x 4 shastry model
'''


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
for i in range(Lx*Ly+1):
    print(i)
    basis_2d = spin_basis_general(N_2d, Nup = i ,pauli=False)
    H_ = quantum_operator(ops_dict,basis=basis_2d,dtype=np.float64, check_symm=False)
    E_,V_ = H_.eigh({})
    E.append(E_)
E = np.concatenate(E,axis=0)

try:
  print("save complete")
  np.save(f"../doc/data/shastry_exact_J1={J1}.npy", E)
except:
  print("specified folder not exsit")
  np.save(f"shastry_exact_J1={J1}.npy", E)
