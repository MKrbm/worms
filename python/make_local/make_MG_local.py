import numpy as np
from scipy import sparse
import os
import sys
sys.path.insert(0, "../") 
from functions import *



I = np.identity(2)

Sz = np.zeros([2,2])
Sz[0,0] = 1/2
Sz[1,1] = -1/2
Sx = np.zeros([2,2])
Sx[1,0] = 1/2
Sx[0,1] = 1/2
Sy = np.zeros([2,2], dtype=np.complex128)
Sy[1,0] = 1j/2
Sy[0,1] = -1j/2


SzSz = np.kron(Sz,Sz).astype(np.float64)
SxSx = np.kron(Sx,Sx).astype(np.float64)
SySy = np.kron(Sy,Sy).astype(np.float64)

lh = SzSz + SxSx + SySy
lh = - lh


LH_ = sparse.csr_matrix((2**3,2**3), dtype = np.float64)
i = 0
LH_ += l2nl(lh/2, 3, [0, 1], sps = 2)
LH_ += l2nl(lh/2, 3, [0, 2], sps = 2)
LH_ += l2nl(lh/2, 3, [1, 2], sps = 2)
print(LH_)


LH = sparse.csr_matrix((2**6,2**6), dtype = np.float64)
LH += l2nl(LH_/2, 6, [0, 1, 2], sps = 2)
LH += l2nl(LH_, 6, [1, 2, 3], sps = 2)
LH += l2nl(LH_, 6, [2, 3, 4], sps = 2)
LH += l2nl(LH_/2, 6, [3, 4, 5], sps = 2)



path = "../array/MG_ori_bond"
if not os.path.isfile(path):
  np.save(path,lh)
  print("save : ", path+".npy")
  beauty_array(lh,path + ".txt")

path = "../array/MG_union_bond"
if not os.path.isfile(path):
  np.save(path,LH.toarray())
  print("save : ", path+".npy")
  beauty_array(LH,path + ".txt")




