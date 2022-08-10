import numpy as np
import sys
sys.path.append('../nsp')
from nsp.utils.func import *



Sz = np.zeros([2,2])
Sz[0,0] = 1/2
Sz[1,1] = -1/2
Sx = np.zeros([2,2])
Sx[1,0] = 1/2
Sx[0,1] = 1/2
Sy = np.zeros([2,2], dtype=np.complex64)
Sy[1,0] = 1j/2
Sy[0,1] = -1j/2


SzSz = np.kron(Sz,Sz).real.astype(np.float64)
SxSx = np.kron(Sx,Sx).real.astype(np.float64)
SySy = np.kron(Sy,Sy).real.astype(np.float64)

lh = SzSz + SxSx + SySy


L = 6
assert L%3 == 0, "L should be 3*K "
bounds = [[0,1], [0, 2], [1, 2]]
lh2 = sum_ham(lh, bounds, 3, 2)
LH = sum_ham(lh2, [[0,1,2], [3, 4, 5]], 6, 2) + sum_ham(lh2/2, [[1, 2, 3], [2, 3, 4]], 6, 2)


bounds = [[i, (i+1)%int(L/3)] for i in range(int(L/3))]

H = sum_ham(LH, [[0, 1], [1, 0]], 2, 8)

print(np.linalg.eigvalsh(H))