import sys,os
import numpy as np
from scipy import sparse

from quspin.basis import spin_basis_1d, spin_basis_general
from quspin.operators import hamiltonian,quantum_operator
from quspin.tools.lanczos import lanczos_full,lanczos_iter,FTLM_static_iteration,LTLM_static_iteration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian,quantum_operator
from quspin.tools.lanczos import lanczos_full,lanczos_iter,FTLM_static_iteration,LTLM_static_iteration
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path+"/../reduce_nsp")
from nsp.utils import save_fig

from six import iteritems

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
    p = np.exp(-np.outer(np.atleast_1d(beta),E))
    c_dict = {key:np.einsum("j,aj,j,...j->a...",V[0,:],V,A,p) for key, A in iteritems(O_dict)}
    c = np.einsum("j,aj,...j->a...",V[0,:],V,p)
    r,Q_T = _get_first_lv(iter(Q_T))
    results_dict = {}
    
#     Ar_dict = {key:A.dot(r) for key,A in iteritems(O_dict)}

    for i,lv in enumerate(Q_T): # nv matvecs
        for key,A in iteritems(O_dict):
            if key in results_dict:
                results_dict[key] += np.squeeze(c_dict[key][i,...] * np.vdot(lv,r))
            else:
                results_dict[key]  = np.squeeze(c_dict[key][i,...] * np.vdot(lv,r))

    return results_dict,np.squeeze(c[0,...])


"""
H : Hamiltonian
M : Magnetization
"""

model_names = [
    "HXXZ1D", # Heisenberg 1D
    "HXXZ2D", # Heisenberg 2D
    "HM2D",
    "SS"
]

def lancoz_estimate(_H, params_dict ,M, N_samples, Kdim, beta, model_name = "HXXZ1D", printout=False):

    printout = printout and (len(beta)==1)
    
    H = _H.tohamiltonian(params_dict)
    basis = _H.basis
    if model_name not in model_names:
        raise ValueError("model_name not in {}".format(model_names))

    npy_path = dir_path + "/npy/" + model_name + "/" 
    img_path = dir_path + "/images/" + model_name + "/"

    a = ""
    for k, v in params_dict.items():
        v = float(v)
        a += f"{k}_{v:.4g}_"
    params_str = a[:-1]

    N_sites = basis.N
    out = np.zeros((Kdim,H.Ns),dtype=np.float64)
    [E0] = H.eigsh(k=1,which="SA",return_eigenvectors=False)
    E_list = []
    Vs = []
    lvs = []
    for i in range(N_samples):
        r = np.random.normal(0,1,size=H.Ns)
        r /= np.linalg.norm(r)
        E,V,lv = lanczos_full(H,r,Kdim,eps=1e-8,full_ortho=True)
        # E -= E0
        E_list.append(E)
        Vs.append(V)
        lvs.append(lv)

    H2 = H**2
    M2 = M**2


    # calculate energy, specificheat, magnetization, and square of magnetization
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
        results_FT,Id_FT = FTLM_static_iteration({ "M":M, "M2":M2},E - E0,V,lv,beta=beta)
        results_LT,Id_LT = LTLM_static_iteration({ "M":M, "M2":M2},E - E0,V,lv,beta=beta)
        results_FT_H ,Id_FT = FTLM_static_iteration_poly({"E":E, "H2":E**2},E - E0,V,lv,beta=beta) # FTLM and LTLM are the same for diagonal operators
        
        #append magnetization and square of magnetization
        M_FT_list.append(results_FT["M"])
        M_LT_list.append(results_LT["M"])
        M2_FT_list.append(results_FT["M2"])
        M2_LT_list.append(results_LT["M2"])


        E_poly_list.append(results_FT_H["E"])
        H2_poly_list.append(results_FT_H["H2"])
        Z_FT_list.append(Id_FT)
        Z_LT_list.append(Id_LT)

    # calculating error bars on the expectation values
    E_poly,dE_poly = bootstrap_mean(E_poly_list,Z_FT_list)
    H2_poly,dH2_poly = bootstrap_mean(H2_poly_list,Z_FT_list)

    # calculating error bars on the magnetization and square of magnetization
    M_FT,dM_FT = bootstrap_mean(M_FT_list,Z_FT_list)
    M_LT,dM_LT = bootstrap_mean(M_LT_list,Z_LT_list)
    M2_FT,dM2_FT = bootstrap_mean(M2_FT_list,Z_FT_list)
    M2_LT,dM2_LT = bootstrap_mean(M2_LT_list,Z_LT_list)

    C_poly = (H2_poly - E_poly**2) * (beta**2)
    dC_poly = dH2_poly * (beta**2)

    T = 1 / beta

    if printout:
        beta = beta[0]
        chi_FT = M2_FT - M_FT**2
        dchi_FT = dM2_FT + dM_FT * M_FT * 2
        chi_LT = M2_LT - M_LT**2
        chi_LT = dM2_LT + dM_LT * M_LT * 2

        print("model name : {}".format(model_name))
        print("N_sites : {}".format(N_sites))
        print("params : {}".format(params_str))
        print("N_samples : {}".format(N_samples))
        print("Kdim : {}".format(Kdim))
        print("---------------------------------")
        print(f"temperature           = {T[0]}")
        print(f"energy(per site)      =  {E_poly / N_sites } += {dE_poly / N_sites}")
        print(f"specific heat         =  {C_poly[0] / N_sites} += {dC_poly[0] / N_sites}")
        print(f"magnetization(FTLM)   =  {M_FT} += {dM_FT}")
        print(f"magnetization(LTLM)   =  {M_LT} += {dM_LT}")
        print(f"suceptibility(FTLM)   =   \
            {chi_FT * (beta) * N_sites} += {dchi_FT * (beta)* N_sites}")
        print(f"suceptibility(LTLM)   =  \
            {chi_LT * (beta) * N_sites} += {chi_LT * (beta)* N_sites}")
        return


    np.save(npy_path + f"E_poly_" + params_str, [E_poly, dE_poly, T])
    np.save(npy_path + f"C_poly_" + params_str, [C_poly, dC_poly, T])

    np.save(npy_path + f"M_FT_" + params_str, [M_FT, dM_FT, T])
    np.save(npy_path + f"M_LT_" + params_str, [M_LT, dM_LT, T])
    np.save(npy_path + f"M2_FT_" + params_str, [M2_FT, dM2_FT, T])
    np.save(npy_path + f"M2_LT_" + params_str, [M2_LT, dM2_LT, T])
    

    ##### plot results #####

    # setting up plot and inset
    fig_size=6 # figure aspect ratio parameter
    fig,ax = plt.subplots(figsize=(1.5*fig_size,fig_size))
    ax.errorbar(T,E_poly,dE_poly,marker=".",label="FTLM_poly",zorder=-2)

    ax.set_xscale("log")
    xmin,xmax = ax.get_xlim()
    ax.legend(loc="lower right")
    ax.set_xlabel("temperature")
    ax.set_ylabel("energy")
    ax.grid(which="both", ls="-")
    save_fig(fig, img_path + f"/" + params_str + f"/ M_{Kdim}_N_{N_samples}", f"E", 400, overwrite = True)


    ##### plot results #####
    #
    # setting up plot and inset
    fig,axs = plt.subplots(ncols=2, figsize=(1.5*fig_size,2*fig_size))
    C = np.gradient(E_poly, T)
    axs[0].plot(T, C, marker='.', label = "derivative of E_poly", zorder=-1)
    axs[0].plot(T,C_poly,marker=".",label="Lancoz poly",zorder=-3)

    axs[0].set_xscale("log")
    xmin,xmax = ax.get_xlim()
    axs[0].legend(loc="lower right")
    axs[0].set_ylabel("energy")
    axs[0].grid(which="both", ls="-")

    axs[1].plot(T, C, marker='.', label = "derivative of E_poly", zorder=-1)
    axs[1].plot(T,C_poly,marker=".",label="Lancoz poly",zorder=-3)

    xmin,xmax = ax.get_xlim()
    axs[1].set_xlabel("temperature")
    axs[1].set_ylabel("energy")
    axs[1].grid(which="both", ls="-")

    fig.tight_layout()
    save_fig(fig, img_path + f"/" + params_str + f"/ M_{Kdim}_N_{N_samples}", f"C", 400, overwrite = True)

    ### plot magnetization ###
    fig,axs = plt.subplots(2, figsize=(1.5*fig_size,2*fig_size))
    axs[0].errorbar(T,M_FT,dM_FT,marker=".",label="FTLM",zorder=-1)
    axs[0].errorbar(T,M_LT,dM_LT,marker=".",label="LTLM",zorder=-2)
    axs[0].set_xscale("log")
    xmin,xmax = ax.get_xlim()
    axs[0].legend(loc="lower right")
    axs[0].set_xlabel("temperature")
    axs[0].set_ylabel("magnetization")
    axs[0].grid(which="both", ls="-")
    axs[1].errorbar(T,M2_FT,dM2_FT,marker=".",label="FTLM",zorder=-1)
    axs[1].errorbar(T,M2_LT,dM2_LT,marker=".",label="LTLM",zorder=-2)
    axs[1].set_xscale("log")
    xmin,xmax = ax.get_xlim()
    axs[1].legend(loc="lower right")
    axs[1].set_xlabel("temperature")
    axs[1].set_ylabel("magnetization^2")
    axs[1].grid(which="both", ls="-")
    fig.tight_layout()
    save_fig(fig, img_path + f"/" + params_str + f"/ M_{Kdim}_N_{N_samples}", f"M", 400, overwrite = True)




    fig,ax = plt.subplots(figsize=(1.5*fig_size,fig_size))
    ax.errorbar(T,M2_FT * (beta) * N_sites , dM2_FT * (beta),marker=".",label="FTLM",zorder=-1)
    ax.errorbar(T,M2_LT * (beta) * N_sites , dM2_LT * (beta),marker=".",label="LTLM",zorder=-2)
    ax.set_xscale("log")
    ax.grid(which="both", ls="-")
    xmin,xmax = ax.get_xlim()
    ax.legend(loc="lower right")
    ax.set_xlabel("temperature")
    ax.set_ylabel("susceptibility")
    ax.set_title(f"susceptibility vs T")
    fig.tight_layout()
    save_fig(fig, img_path + f"/" + params_str + f"/ M_{Kdim}_N_{N_samples}", f"kai", 400, overwrite = True)


