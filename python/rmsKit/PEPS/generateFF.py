import numpy as np
import sys
import tensornetwork as tn
import numpy as np

sys.path.append("../")
import rms_torch


def block(*dimensions):
    """Construct a new matrix for the MPS with random numbers from 0 to 1"""
    size = tuple([x for x in dimensions])
    return np.random.random_sample(size)


def create_MPS(rank, dimension, bond_dimension):
    """Build the MPS tensor"""
    A = block(bond_dim, dim, bond_dim)
    mps = [tn.Node(np.copy(A)) for _ in range(rank)]
    # connect edges to build mps
    connected_edges = []
    for k in range(0, rank):
        conn = mps[k][2] ^ mps[(k + 1) % rank][0]
        connected_edges.append(conn)

    return mps, connected_edges


dim = s = 3
bond_dim = 2
rank = L = 6
mps_nodes, mps_edges = create_MPS(rank, dim, bond_dim)
for k in range(len(mps_edges)):
    A = tn.contract(mps_edges[k])

y = A.tensor.reshape(-1)
rho = y[:, None] @ y[None, :]
rho_ = rho.reshape(s**2, s**(L-2), s**2, s**(L-2))
prho = np.einsum("jiki->jk", rho_)
e, V = np.linalg.eigh(prho)
e = np.round(e, 10)
P = np.diagflat((e == 0)).astype(np.float64)
vp = V @ P 
h = vp @ vp.T
bonds = [[i, (i+1)%L] for i in range(L)]
H = rms_torch.sum_ham(h, bonds, L, s)

print(np.linalg.eigvalsh(H))
