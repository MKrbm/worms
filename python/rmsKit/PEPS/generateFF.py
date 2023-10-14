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

s = args.sps
bond_dim = args.rank
n = args.length
d = args.dimension

def block(*dimensions):
    """Construct a new matrix for the MPS with random numbers from 0 to 1"""
    size = tuple([x for x in dimensions])
    return np.random.random_sample(size)


def create_MPS(n, dimension, bd):
    """Build the MPS tensor"""
    A = block(bond_dim, s, bond_dim)
    mps = [tn.Node(np.copy(A)) for _ in range(n)]
    # connect edges to build mps
    connected_edges = []
    for k in range(0, n):
        conn = mps[k][2] ^ mps[(k + 1) % n][0]
        connected_edges.append(conn)

    return mps, connected_edges


mps_nodes, mps_edges = create_MPS(n, s, bond_dim)
for k in range(len(mps_edges)):
    A = tn.contract(mps_edges[k])

y = A.tensor.reshape(-1)
rho = y[:, None] @ y[None, :]
rho_ = rho.reshape(s**2, s ** (n - 2), s**2, s ** (n - 2))
prho = np.einsum("jiki->jk", rho_)
e, V = np.linalg.eigh(prho)
e = np.round(e, 10)
P = np.diagflat((e == 0)).astype(np.float64)
vp = V @ P
h = vp @ vp.T
bonds = [[i, (i + 1) % n] for i in range(n)]

path = f"../array/torch/ff/s_{s}_bd_{bond_dim}_n{n}_d{d}"

H = utils.sum_ham(h, bonds, n, s)
save_npy(f"{path}/H", [h])
