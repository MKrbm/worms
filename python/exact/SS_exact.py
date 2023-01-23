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


sys.path.insert(0, "/home/user/project/python/reduce_nsp")
from nsp.utils import save_fig


os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='12' # set number of MKL threads to run in parallel
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
parser.add_argument('-H','--magnetic', help='magnetic field', type = float, default = 0)
parser.add_argument('-n','--n_samples', help='# of samples', type = int, default = 50)
parser.add_argument('-m','--n_Krylov', help='dimernsion of Krylov space', type = int, default = 50)


args = parser.parse_args()

'''
calculate exact 4 x 4 shastry model
'''
def bootstrap_mean(O_r,Id_r,n_bootstrap=100):
    """
    Uses boostraping to esimate the error due to sampling.

    O_r: numerator
    Id_r: denominator
    n_bootstrap: bootstrap sample size

    """
    O_r = np.asarray(O_r)
    Id_r = np.asarray(Id_r)
    #
    avg = np.nanmean(O_r,axis=0)/np.nanmean(Id_r,axis=0)
    n_Id = Id_r.shape[0]
    #n_N = O_r.shape[0]
    #
    i_iter = (np.random.randint(n_Id,size=n_Id) for i in range(n_bootstrap))
    #
    bootstrap_iter = (np.nanmean(O_r[i,...],axis=0)/np.nanmean(Id_r[i,...],axis=0) for i in i_iter)
    diff_iter = ((bootstrap-avg)**2 for bootstrap in bootstrap_iter)
    err = np.sqrt(sum(diff_iter)/n_bootstrap)
    #
    return avg,err


from scipy.linalg import eigh_tridiagonal
from six import iteritems
import numpy as _np
def _get_first_lv_iter(r,Q_iter):
    yield r
    for Q in Q_iter:
        yield Q


def _get_first_lv(Q_iter):
    r = next(Q_iter)
    return r,_get_first_lv_iter(r,Q_iter)

# FTLM and LTLM reduce to same algorithm if O_dict is diagonal in eigenvectors.
def FTLM_static_iteration_poly(O_dict,E,V,Q_T,beta=0):
    # O_dict can contains operators polynomial in H
    p = _np.exp(-_np.outer(_np.atleast_1d(beta),E))
    c_dict = {key:_np.einsum("j,aj,j,...j->a...",V[0,:],V,A,p) for key, A in iteritems(O_dict)}
    c = _np.einsum("j,aj,...j->a...",V[0,:],V,p)
    r,Q_T = _get_first_lv(iter(Q_T))
    results_dict = {}
    
#     Ar_dict = {key:A.dot(r) for key,A in iteritems(O_dict)}

    for i,lv in enumerate(Q_T): # nv matvecs
        for key,A in iteritems(O_dict):
            if key in results_dict:
                results_dict[key] += _np.squeeze(c_dict[key][i,...] * _np.vdot(lv,r))
            else:
                results_dict[key]  = _np.squeeze(c_dict[key][i,...] * _np.vdot(lv,r))

    return results_dict,_np.squeeze(c[0,...])


Lx, Ly = 4, 4 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction


# set up coupling 
J1 = args.coupling # J2 represent coupling constant between diagonal part (if J2 strong (or J1 week) dimer appears)
h = args.magnetic
J_zz_p = [[1,i,T_x[i]] for i in range(N_2d)]+[[1,i,T_y[i]] for i in range(N_2d)] 
J_zz_x = [[1,0,5],[1,2,7],[1,8,13],[1,10,15],[1,1,14],[1,3,12],[1,6,9],[1,4,11]]
J_xy_p = [[1/2.0,i,T_x[i]] for i in range(N_2d)]+[[1/2.0,i,T_y[i]] for i in range(N_2d)]
J_xy_x = [[1/2.0,0,5],[1/2.0,2,7],[1/2.0,8,13],[1/2.0,10,15],[1/2.0,1,14],[1/2.0,3,12],[1/2.0,6,9],[1/2.0,4,11]]
h_list = [[-1.0,i] for i in range(N_2d)]
ops_dict = dict(Jp=[["+-",J_xy_p], ["-+",J_xy_p], ["zz", J_zz_p]],
                Jx=[["+-",J_xy_x], ["-+",J_xy_x], ["zz", J_zz_x]],
                h=[["z",h_list]])
basis_2d = spin_basis_general(N_2d ,pauli=False)
H_SS = quantum_operator(ops_dict,basis=basis_2d,dtype=np.float64, check_symm=False)

# define magnetization observable
M_list = [[1.0/N_2d,i] for i in range(N_2d)]
M = hamiltonian([["z",M_list]],[],basis=basis_2d,dtype=np.float64)
M2 = M**2


# define hamiltonian for given parameters
params_dict=dict(Jp=J1,h=h)
H = H_SS.tohamiltonian(params_dict)



E = []
L = N_2d # system size
N_samples = args.n_samples # of samples to approximate thermal expectation value with
m = args.n_Krylov # dimensio of Krylov space
out = np.zeros((m,H.Ns),dtype=np.float64)
[E0] = H.eigsh(k=1,which="SA",return_eigenvectors=False)
E_list = []
Vs = []
lvs = []
for i in range(N_samples):
    r = np.random.normal(0,1,size=H.Ns)
    r /= np.linalg.norm(r)
    E,V,lv = lanczos_full(H,r,m,eps=1e-8,full_ortho=True)
    # E -= E0
    E_list.append(E)
    Vs.append(V)
    lvs.append(lv)

# np.save(f"npy/SS_GS_J_{J1}_h{h}", E0)
# np.save(f"npy/SS_E_J_{J1}_h{h}_N_{m}x{N_samples}", E_list)
# np.save(f"npy/SS_V_J_{J1}_h{h}_N_{m}x{N_samples}", Vs)
# np.save(f"npy/SS_lv_J_{J1}_h{h}_N_{m}x{N_samples}", lvs)

# import pickle
# with open(f'npy/SS_H_J_{J1}.pickle', 'wb') as handle:
#     pickle.dump(H, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("done calculation ")

H2 = H.tocsc()**2

#* for loading 
"""
Es = np.load("npy/SS_E_J_1_N_50x50.npy")
Vs = np.load("npy/SS_V_J_1_N_50x50.npy")
lvs = np.load("npy/SS_lv_J_1_N_50x50.npy")
E0 = np.load("npy/SS_GS_J_1.npy")

with open('npy/SS_H_J_1.pickle', 'rb') as handle:
    H = pickle.load(handle)
"""



# calculate energy, specificheat, magnetization, and square of magnetization

E_FT_list = []
E_LT_list = []
H2_FT_list = []
H2_LT_list = []
Z_FT_list = []
Z_LT_list = []
E_poly_list = []
H2_poly_list = []


M_FT_list = []
M_LT_list = []
M2_FT_list = []
M2_LT_list = []

T = np.logspace(-1.6, 0.3, num=80) # temperature vector
beta = 1.0/(T+1e-15) # inverse temperature vector


for i in range(len(lvs)):
    E,V,lv = E_list[i], Vs[i], lvs[i]
    results_FT,Id_FT = FTLM_static_iteration({ "M":M, "M2":M2},E - E0,V,lv,beta=beta)
    results_LT,Id_LT = LTLM_static_iteration({ "M":M, "M2":M2},E - E0,V,lv,beta=beta)
    results_FT_H ,Id_FT = FTLM_static_iteration_poly({"E":E, "H2":E**2},E - E0,V,lv,beta=beta)
    
    #append magnetization and square of magnetization
    M_FT_list.append(results_FT["M"])
    M_LT_list.append(results_LT["M"])
    M2_FT_list.append(results_FT["M2"])
    M2_LT_list.append(results_LT["M2"])


    # save results to a list
    # E_FT_list.append(results_FT["E"])
    E_poly_list.append(results_FT_H["E"])
    # H2_FT_list.append(results_FT["H2"])
    H2_poly_list.append(results_FT_H["H2"])
    Z_FT_list.append(Id_FT)
    # E_LT_list.append(results_LT["E"])
    # H2_LT_list.append(results_LT["H2"])
    Z_LT_list.append(Id_LT)

# calculating error bars on the expectation values
# E_FT,dE_FT = bootstrap_mean(E_FT_list,Z_FT_list)
E_poly,dE_poly = bootstrap_mean(E_poly_list,Z_FT_list)
# E_LT,dE_LT = bootstrap_mean(E_LT_list,Z_LT_list)

# H2_FT,dH2_FT = bootstrap_mean(H2_FT_list,Z_FT_list)
H2_poly,dH2_poly = bootstrap_mean(H2_poly_list,Z_FT_list)
# H2_LT,dH2_LT = bootstrap_mean(H2_LT_list,Z_LT_list)

# calculating error bars on the magnetization and square of magnetization
M_FT,dM_FT = bootstrap_mean(M_FT_list,Z_FT_list)
M_LT,dM_LT = bootstrap_mean(M_LT_list,Z_LT_list)
M2_FT,dM2_FT = bootstrap_mean(M2_FT_list,Z_FT_list)
M2_LT,dM2_LT = bootstrap_mean(M2_LT_list,Z_LT_list)



C_poly = (H2_poly - E_poly**2) * (beta**2)

# dC_FT = dH2_FT * (beta**2)
# dC_LT = dH2_LT * (beta**2)
dC_poly = dH2_poly * (beta**2)

# np.save(f"npy/SS_E_FT_J_{J1}", [E_FT, T])
# np.save(f"npy/SS_E_LT_J_{J1}", [E_LT, T])

np.save(f"npy/SS_E_poly_J_{J1}_h_{h}", [E_poly, T])

# np.save(f"npy/SS_C_FT_J_{J1}", [C_FT, T])
# np.save(f"npy/SS_C_LT_J_{J1}", [C_LT, T])
np.save(f"npy/SS_C_poly_J_{J1}_h_{h}", [C_poly, T])

np.save(f"npy/SS_M_FT_J_{J1}_h_{h}", [M_FT, T])
np.save(f"npy/SS_M_LT_J_{J1}_h_{h}", [M_LT, T])
np.save(f"npy/SS_M2_FT_J_{J1}_h_{h}", [M2_FT, T])
np.save(f"npy/SS_M2_LT_J_{J1}_h_{h}", [M2_LT, T])


##### plot results #####

# setting up plot and inset
fig_size=6 # figure aspect ratio parameter
fig,ax = plt.subplots(figsize=(1.5*fig_size,fig_size))
# ax.errorbar(T,E_LT,dE_LT,marker=".",label="LTLM",zorder=-1)
# ax.errorbar(T,E_FT,dE_FT,marker=".",label="FTLM",zorder=-2)
ax.errorbar(T,E_poly,dE_poly,marker=".",label="FTLM_poly",zorder=-2)

ax.set_xscale("log")
xmin,xmax = ax.get_xlim()
ax.legend(loc="lower right")
ax.set_xlabel("temperature")
ax.set_ylabel("energy")

save_fig(fig,f"images/SS/J_{J1}_h_{h}/M_{m}_N_{N_samples}", f"E", 400, overwrite = True)


##### plot results #####
#
# setting up plot and inset
fig,axs = plt.subplots(2, figsize=(1.5*fig_size,2*fig_size))
C = np.gradient(E_poly, T)
axs[0].plot(T, C, marker='.', label = "derivative of E_poly", zorder=-1)
axs[0].plot(T,C_poly,marker=".",label="FTLM poly",zorder=-3)

axs[0].set_xscale("log")
xmin,xmax = ax.get_xlim()
axs[0].legend(loc="lower right")
axs[0].set_ylabel("energy")


axs[1].plot(T, C, marker='.', label = "derivative of E_FT", zorder=-1)
axs[1].plot(T,C_poly,marker=".",label="FTLM poly",zorder=-3)

xmin,xmax = ax.get_xlim()
axs[1].set_xlabel("temperature")
axs[1].set_ylabel("energy")

fig.tight_layout()
save_fig(fig,f"images/SS/J_{J1}_h_{h}/M_{m}_N_{N_samples}", f"C", 400, overwrite = True)

### plot magnetization ###
fig,axs = plt.subplots(2, figsize=(1.5*fig_size,2*fig_size))
axs[0].errorbar(T,M_FT,dM_FT,marker=".",label="FTLM",zorder=-1)
axs[0].errorbar(T,M_LT,dM_LT,marker=".",label="LTLM",zorder=-2)
axs[0].set_xscale("log")
xmin,xmax = ax.get_xlim()
axs[0].legend(loc="lower right")
axs[0].set_xlabel("temperature")
axs[0].set_ylabel("magnetization")

axs[1].errorbar(T,M2_FT,dM2_FT,marker=".",label="FTLM",zorder=-1)
axs[1].errorbar(T,M2_LT,dM2_LT,marker=".",label="LTLM",zorder=-2)
axs[1].set_xscale("log")
xmin,xmax = ax.get_xlim()
axs[1].legend(loc="lower right")
axs[1].set_xlabel("temperature")
axs[1].set_ylabel("magnetization^2")

fig.tight_layout()
save_fig(fig,f"images/SS/J_{J1}_h_{h}/M_{m}_N_{N_samples}", f"M", 400, overwrite = True)




# calculate susceptibility by simulate h = 0.05 * J1
"""
* not need to calculate by numerical differentiation
h_prime = 0.1 * J1 + h

# define hamiltonian for given parameters
params_dict=dict(Jp=J1,h=h_prime)
H = H_SS.tohamiltonian(params_dict)



E = []
L = N_2d # system size
out = np.zeros((m,H.Ns),dtype=np.float64)
[E0] = H.eigsh(k=1,which="SA",return_eigenvectors=False)
E_list = []
Vs = []
lvs = []
for i in range(N_samples):
    r = np.random.normal(0,1,size=H.Ns)
    r /= np.linalg.norm(r)
    E,V,lv = lanczos_full(H,r,m,eps=1e-8,full_ortho=True)
    # E -= E0
    E_list.append(E)
    Vs.append(V)
    lvs.append(lv)

# np.save(f"npy/SS_GS_J_{J1}_h{h_prime}", E0)
# np.save(f"npy/SS_E_J_{J1}_h{h_prime}_N_{m}x{N_samples}", E_list)
# np.save(f"npy/SS_V_J_{J1}_h{h_prime}_N_{m}x{N_samples}", Vs)
# np.save(f"npy/SS_lv_J_{J1}_h{h_prime}_N_{m}x{N_samples}", lvs)



#* for loading 
# Es = np.load("npy/SS_E_J_1_N_50x50.npy")
# Vs = np.load("npy/SS_V_J_1_N_50x50.npy")
# lvs = np.load("npy/SS_lv_J_1_N_50x50.npy")
# E0 = np.load("npy/SS_GS_J_1.npy")

# with open('npy/SS_H_J_1.pickle', 'rb') as handle:
#     H = pickle.load(handle)



# calculate energy, specificheat, magnetization, and square of magnetization

E_FT_list = []
E_LT_list = []
H2_FT_list = []
H2_LT_list = []
Z_FT_list = []
Z_LT_list = []
E_poly_list = []
H2_poly_list = []


M_FT_list = []
M_LT_list = []
M2_FT_list = []
M2_LT_list = []



for i in range(len(lvs)):
    E,V,lv = E_list[i], Vs[i], lvs[i]
    results_FT,Id_FT = FTLM_static_iteration({ "M":M},E - E0,V,lv,beta=beta)
    results_LT,Id_LT = LTLM_static_iteration({ "M":M},E - E0,V,lv,beta=beta)
    
    #append magnetization and square of magnetization
    M_FT_list.append(results_FT["M"])
    M_LT_list.append(results_LT["M"])

    Z_FT_list.append(Id_FT)
    Z_LT_list.append(Id_LT)

M_FT_h,dM_FT_h = bootstrap_mean(M_FT_list,Z_FT_list)
M_LT_h,dM_LT_h = bootstrap_mean(M_LT_list,Z_LT_list)

kai_FL_h = M_FT_h / h_prime
kai_LT_h = M_LT_h / h_prime
dka_FL_h = dM_FT_h / h_prime
dkai_LT_h = dM_LT_h / h_prime
"""



##### plot results susceptibility #####

# setting up plot and inset
fig,ax = plt.subplots(figsize=(1.5*fig_size,fig_size))
# axs[0].errorbar(T,kai_FL_h,dka_FL_h,marker=".",label="FTLM diff",zorder=-1)
# axs[0].errorbar(T,kai_LT_h,dkai_LT_h,marker=".",label="LTLM diff",zorder=-2)

# axs[0].set_xscale("log")
# xmin,xmax = axs[0].get_xlim()
# axs[0].legend(loc="lower right")
# axs[0].set_xlabel("temperature")
# axs[0].set_ylabel(f"susceptibility")
# axs[0].set_title(f"susceptibility (calculated by numerical differentiation dh = {h_prime - h }) vs T")


ax.errorbar(T,M2_FT * (beta) * N_2d , dM2_FT * (-beta),marker=".",label="FTLM",zorder=-1)
ax.errorbar(T,M2_LT * (beta) * N_2d , dM2_LT * (-beta),marker=".",label="LTLM",zorder=-2)
ax.set_xscale("log")
xmin,xmax = ax.get_xlim()
ax.legend(loc="lower right")
ax.set_xlabel("temperature")
ax.set_ylabel("susceptibility")
ax.set_title(f"susceptibility vs T")
fig.tight_layout()
save_fig(fig,f"images/SS/J_{J1}_h_{h}/M_{m}_N_{N_samples}", f"kai", 400, overwrite = True)


