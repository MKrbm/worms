import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import sparse
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

SzSz = sparse.kron(Sz,Sz,format='csr').real
SxSx = sparse.kron(Sx,Sx,format='csr').real
SySy = sparse.kron(Sy,Sy,format='csr').real

path1 = "array/H_bond_z"
if not os.path.isfile(path1):
  np.save(path1,SzSz.toarray())
  print("save : ", path1+".npy")
  beauty_array(SzSz,path1 + ".txt")


path2 = "array/H_bond_x"
if not os.path.isfile(path2):
  np.save(path2,SxSx.toarray())
  print("save : ", path2+".npy")
  beauty_array(SxSx, path2 + ".txt")

path3 = "array/H_bond_y"
if not os.path.isfile(path3):
  np.save(path3,SySy.toarray())
  print("save : ", path3+".npy")
  beauty_array(SySy,path3 + ".txt")