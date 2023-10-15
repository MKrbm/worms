# This code will generate the local hamiltonian for d-dimensional square lattice that satisfy frustraion free.
import numpy as np
import sys
import tensornetwork as tn
sys.path.append("../")
from lattice import save_npy, list_unitaries
import rms_torch
import utils
import argparse

import logging
import datetime

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"log/{now}_optimizer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_filename,
    # handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
print(f"logging to file: {log_filename}")

parser = argparse.ArgumentParser(
    description="This code will generate the local hamiltonian for d-dimensional square lattice that satisfy frustraion free."
)
parser.add_argument(
    "-s",
    "--sps",
    help="degree of spin freedom",
    required=True,
    type=int,
)
parser.add_argument("-r", "--rank", help="rank of bond", default=2, type=int)
parser.add_argument("-l", "--length", help="number of spin", default=6, type=int)
parser.add_argument("-d", "--dimension", help="dimension of lattice", default=1, type=int)
args = parser.parse_args()

sps = args.sps
bond_dim = args.rank
n = args.length
d = args.dimension

def block(*dimensions):
    '''Construct a new matrix for the MPS with random numbers from 0 to 1'''
    size = tuple([x for x in dimensions])
    A = np.random.random_sample(size)
    A = ((A.transpose(2,1,0) + A ) / 2)
    return A


def create_MPS(n, dimension, bd, A):
    """Build the MPS tensor"""
    mps = [tn.Node(np.copy(A)) for _ in range(n)]
    # connect edges to build mps
    connected_edges = []
    for k in range(0, n):
        conn = mps[k][2] ^ mps[(k + 1) % n][0]
        connected_edges.append(conn)

    return mps, connected_edges

A = block(bond_dim, sps, bond_dim)

#Calculate the kernel of A - A 
A2 = np.einsum("ijk,klm->jlim", A, A).reshape(sps**2, bond_dim**2)
U, s, V = np.linalg.svd(A2)
Up = U[:, len(s):]
h = Up @ Up.T

mps_nodes, mps_edges = create_MPS(n, sps, bond_dim, A)
for k in range(len(mps_edges)):
    A = tn.contract(mps_edges[k])

yL = A.tensor.reshape(sps*sps,-1)
yL /= np.linalg.norm(yL)
# logging.info(np.linalg.norm(h @ yL))
logging.info("Confirm this is indeed a frustration free hamiltonian h @ yL = 0 : %s", np.linalg.norm(h @ yL))
path = f"../array/torch/ff/s_{sps}_bd_{bond_dim}_n{n}_d{d}"
save_npy(f"{path}/H", [h])
