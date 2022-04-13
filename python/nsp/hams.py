import sys
sys.path.append('..')

import numpy as np
import functions as f
from importlib import reload
from scipy import sparse
import scipy.sparse.linalg
import scipy


L = 6

Sz = np.zeros([2,2])
Sz[0,0] = 1/2
Sz[1,1] = -1/2
Sx = np.zeros([2,2])
Sx[1,0] = 1/2
Sx[0,1] = 1/2
Sy = np.zeros([2,2], dtype=np.complex64)
Sy[1,0] = 1j/2
Sy[0,1] = -1j/2


SzSz = np.kron(Sz,Sz).astype(np.float64)
SxSx = np.kron(Sx,Sx).astype(np.float64)
SySy = np.kron(Sy,Sy).astype(np.float64)

lh = SzSz + SxSx + SySy

H = sparse.csr_matrix((2**L, 2**L), dtype=np.float64)
for i in range(L):
    
    H += f.l2nl(lh, L, [i,(i+1)%L], sps = 2)
    H += f.l2nl(lh/2, L, [i,(i+2)%L], sps = 2)   

E_MG = np.linalg.eigvalsh(H.toarray())

LH_ = sparse.csr_matrix((2**3,2**3), dtype = np.float64)
i = 0
LH_ += f.l2nl(lh/2, 3, [0, 1], sps = 2)
LH_ += f.l2nl(lh/2, 3, [0, 2], sps = 2)
LH_ += f.l2nl(lh/2, 3, [1, 2], sps = 2)

LH = sparse.csr_matrix((2**6,2**6), dtype = np.float64)
LH += f.l2nl(LH_/2, 6, [0, 1, 2], sps = 2)
LH += f.l2nl(LH_, 6, [1, 2, 3], sps = 2)
LH += f.l2nl(LH_, 6, [2, 3, 4], sps = 2)
LH += f.l2nl(LH_/2, 6, [3, 4, 5], sps = 2)

H = sparse.csr_matrix((2**6,2**6), dtype = np.float64)
H += f.l2nl(LH, 2, [0, 1], sps = 8)
H += f.l2nl(LH, 2, [1, 0], sps = 8)
LH2 = f.l2nl(LH, 2, [1, 0], sps = 8)
X = -H.toarray()
X1 = -LH.toarray()
X2 = -LH2.toarray()
# np.save("../doc/data/MG_exact.npy")