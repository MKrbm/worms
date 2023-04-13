from lattice import KH, HXYZ
from lattice.core.utils import *
import numpy as np
import jax
import jax.numpy as jnp
import os
import argparse

models = [
    "KH",
    "HXYZ"
]
parser = argparse.ArgumentParser(description='exact diagonalization of shastry_surtherland')
parser.add_argument('-m','--model', help='model (model) Name', required=True, choices=models)
parser.add_argument('-Jz','--coupling_z', help='coupling constant (Jz)', type = float, default = 1) # SxSx + SySy + 
parser.add_argument('-Jx','--coupling_x', help='coupling constant (Jx)', type = float) 
parser.add_argument('-Jy','--coupling_y', help='coupling constant (Jy)', type = float) 
parser.add_argument("-hx", "--mag_x", help="magnetic field", type=float, default=0)
parser.add_argument("-hz", "--mag_z", help="magnetic field", type=float, default=0)
parser.add_argument('-T', "--temperature", help = "temperature", type = float)
parser.add_argument('-L1', "--length1", help = "length of side", type = int, required = True)
parser.add_argument('-L2', "--length2", help = "length of side", type = int)
parser.add_argument(
    "-u",
    "--unitary_algorithm",
    help="algorithm determine local unitary matrix",
    default="original",
)
parser.add_argument('-gs', action='store_true', help = "calculate only ground state")

# parser.add_argument('-N','--n_samples', help='# of samples', type = int, default = 50)
# parser.add_argument('-m','--n_Krylov', help='dimernsion of Krylov space', type = int, default = 50)
if __name__ == '__main__':

    args = parser.parse_args()
    L1 = args.length1
    L2 = args.length2 if args.length2 is not None else L1
    p = dict(
        Jx=args.coupling_x if args.coupling_x is not None else args.coupling_z,
        Jy=args.coupling_y if args.coupling_y is not None else args.coupling_z,
        Jz=args.coupling_z,
        hx=args.mag_x,
        hz=args.mag_z,
    )
    a = ""
    for k, v in p.items():
        v = float(v)
        a += f"{k}_{v:.4g}_"
    params_str = a[:-1]
    ua = args.unitary_algorithm

    if (args.model == "KH"):
        model_name = "KH" + f"_{L1}x{L2}"
        path = f"out/{model_name}/{ua}/{params_str}"
        N = L1 * L2 * 3
        H = KH.system([L1, L2], ua, p)
    elif (args.model == "HXYZ"):
        model_name = "KXYZ" + f"_{L1}"
        path = f"out/{model_name}/{ua}/{params_str}"
        N = L1
        H = HXYZ.system([L1], ua, p)
    else:
        raise ValueError("model not found")
    E, V = jnp.linalg.eigh(H)
    file = f'{path}/groundstate.npy'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    save_npy(file, V[:,0])
    file = f'{path}/groundstate.csv'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as dat_file:  
        dat_file.write("index, value\n")
        for i, v in enumerate(V[:,0]):
            dat_file.write(f"{i}, {v:.60g}\n")

    if args.gs:
        exit()
        
    beta = jax.numpy.linspace(0, 10, 1001).reshape(1,-1)
    B = jax.numpy.exp(-beta*E[:,None])
    Z = B.sum(axis=0)
    E_mean = (E[:,None]*B).sum(axis=0) / Z
    E_square_mean = ((E*E)[:,None]*B).sum(axis=0) / Z
    beta = beta.reshape(-1)
    C = (E_square_mean - E_mean**2)*(beta**2)


    # * save calculated data
    file = f'{path}/eigenvalues.npy'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    save_npy(file, E)

    file = f'{path}/eigenvalues.csv'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as dat_file:  
        dat_file.write("index, value\n")
        for i, e in enumerate(E):
            dat_file.write(f"{i}, {e:.60g}\n")



    

    file = f'{path}/statistics.csv'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as dat_file:
        dat_file.write("beta, energy_per_site, specific_heat\n")
        for b, e, c, in zip(beta, E_mean, C):
            dat_file.write(f"{b}, {e/N}, {c/N}\n")
    

      