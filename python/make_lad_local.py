import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg
from functions import *
import os

I = np.identity(2)
Sz = np.zeros([2,2])
Sz[0,0] = 1/2
Sz[1,1] = -1/2
Sx = np.zeros([2,2])
Sx[1,0] = 1/2
Sx[0,1] = 1/2
Sy = np.zeros([2,2], dtype=np.complex64)
Sy[1,0] = 1j/2
Sy[0,1] = -1j/2


Sz = sparse.csr_matrix(Sz)
Sx = sparse.csr_matrix(Sx)
Sy = sparse.csr_matrix(Sy)
I = sparse.csr_matrix(I)

Sz1 = sparse.kron(I,Sz,format='csr')
Sz2 = sparse.kron(Sz,I,format='csr')
SzSz = sparse.kron(Sz,Sz,format='csr')

Sx1 = sparse.kron(I,Sx,format='csr')
Sx2 = sparse.kron(Sx,I,format='csr')
SxSx = sparse.kron(Sx,Sx,format='csr')

Sy1 = sparse.kron(I,Sy,format='csr')
Sy2 = sparse.kron(Sy,I,format='csr')
SySy = sparse.kron(Sy,Sy,format='csr')


h = SzSz + SxSx + SySy

Tz = Sz1+Sz2
Tx = Sx1+Sx2
Ty = Sy1+Sy2

Dz = Sz1-Sz2
Dx = Sx1-Sx2
Dy = Sy1-Sy2

TzTz = sparse.kron(Tz,Tz,format='csr')
TxTx = sparse.kron(Tx,Tx,format='csr')
TyTy = sparse.kron(Ty,Ty,format='csr')

DzDz = sparse.kron(Dz,Dz,format='csr')
DxDx = sparse.kron(Dx,Dx,format='csr')
DyDy = sparse.kron(Dy,Dy,format='csr')

Tz2 = Tz @ Tz + Tx@Tx + Ty@Ty
h1 = TzTz + TxTx + TyTy
h2 = DzDz + DxDx + DyDy
h0 = Tz2/2 - 3/4*sparse.identity(4)

h_on = sparse.kron(sparse.identity(4), h0) + sparse.kron(h0, sparse.identity(4))
h_on /= 2

h1 = -h1.real
h2 = -h2.real
h_on = -h_on.real

path1 = "array/lad_bond_ori0"
if not os.path.isfile(path1):
  np.save(path1,h_on.toarray())
  print("save : ", path1+".npy")
  beauty_array(h_on,path1 + ".txt")

path2 = "array/lad_bond_ori1"
if not os.path.isfile(path2):
  np.save(path2,h1.toarray())
  print("save : ", path2+".npy")
  beauty_array(h1,path2 + ".txt")

path3 = "array/lad_bond_ori2"
if not os.path.isfile(path3):
  np.save(path3,h2.toarray())
  print("save : ", path3+".npy")
  beauty_array(h2, path3 + ".txt")



u = np.array([
    [0,1,0,0],
    [1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [-1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [0,0,0,1]
])

u = sparse.csr_matrix(u)
U = sparse.kron(u, u,format='csr')

h1 = U.T @ h1 @ U
h2 = U.T @ h2 @ U
h_on = U.T @ h_on @ U


path1 = "array/lad_bond_0"
if not os.path.isfile(path1):
  np.save(path1,h_on.toarray())
  print("save : ", path1+".npy")
  beauty_array(h_on,path1 + ".txt")

path2 = "array/lad_bond_1"
if not os.path.isfile(path2):
  np.save(path2,h1.toarray())
  print("save : ", path2+".npy")
  beauty_array(h1,path2 + ".txt")

path3 = "array/lad_bond_2"
if not os.path.isfile(path3):
  np.save(path3,h2.toarray())
  print("save : ", path3+".npy")
  beauty_array(h2, path3 + ".txt")