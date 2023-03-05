from jax_exact import KH
import jax
# from jax import numpy as jnp
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
parser.add_argument('-H','--magnetic', help='magnetic field', type = float, default = 0)
parser.add_argument('-T', "--temperature", help = "temperature", type = float)
parser.add_argument('-L1', "--length1", help = "length of side", type = int, required = True)
parser.add_argument('-L2', "--length2", help = "length of side", type = int)

# parser.add_argument('-N','--n_samples', help='# of samples', type = int, default = 50)
# parser.add_argument('-m','--n_Krylov', help='dimernsion of Krylov space', type = int, default = 50)
if __name__ == '__main__':
    args = parser.parse_args()
    L1 = args.length1
    L2 = args.length2 if args.length2 is not None else L1
    p = dict(
        Jx = args.coupling_x if args.coupling_x is not None else args.coupling_z,
        Jy = args.coupling_y if args.coupling_y is not None else args.coupling_z,
        Jz = args.coupling_z,
        h = args.magnetic,
    )
    if (args.model == "KH"):
        N = L1 * L2 * 3
        H = KH([L1, L2], p)
        print("defined Hamiltonian")
        E, V = jax.scipy.linalg.eigh(H)
        # print(E[])
        print("E0-E10 = ", E[:10])
        print("E0/NJ = ", E[0]/(p["Jz"] * L1 * L2 * 3))
        print("Gap = ", E[1] - E[0])
        
        # calculate the energy
        # T = jax.numpy.arange(0.1, 1, 0.1).reshape(1,-1)

        beta = jax.numpy.linspace(0, 10, 100).reshape(1,-1)
        B = jax.numpy.exp(-beta*E[:,None])
        Z = B.sum(axis=0)
        E_mean = (E[:,None]*B).sum(axis=0) / Z
        E_square_mean = ((E*E)[:,None]*B).sum(axis=0) / Z
        beta = beta.reshape(-1)
        for b, e, c, in zip(beta, E_mean, (E_square_mean - E_mean**2)*(beta**2)):
            print(f"{b} : {e} {c}")

        # print(f"beta            = {beta}")
        # print(f"E               = {E_mean}")
        # print(f"C               = {(E_square_mean - E_mean**2)*(beta**2)}")


      