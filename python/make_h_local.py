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

SzSz = sparse.kron(Sz,Sz,format='csr').real.toarray()
SxSx = sparse.kron(Sx,Sx,format='csr').real.toarray()
SySy = sparse.kron(Sy,Sy,format='csr').real.toarray()
m = sparse.kron(Sz,I,format='csr').real + sparse.kron(I,Sz,format='csr').real
m = m.toarray()


p = np.random.rand(4).reshape(2,2)
P = np.kron(p,p)


path = "array/H_bond_z"
if not os.path.isfile(path):
  np.save(path,-SzSz.astype(np.float64))
  print("save : ", path+".npy")
  beauty_array(-SzSz,path + ".txt")


path = "array/H_bond_x"
if not os.path.isfile(path):
  np.save(path,-SxSx.astype(np.float64))
  print("save : ", path+".npy")
  beauty_array(-SxSx, path + ".txt")

path = "array/H_bond_y"
if not os.path.isfile(path):
  np.save(path,-SySy.astype(np.float64))
  print("save : ", path+".npy")
  beauty_array(-SySy,path + ".txt")

path = "array/H_onsite"
if not os.path.isfile(path):
  np.save(path,m.astype(np.float64))
  print("save : ", path+".npy")
  beauty_array(m, path + ".txt")



path = "array/HP_bond_z"
h = P @ (-SzSz) @ np.linalg.inv(P)
if not os.path.isfile(path):
  np.save(path,h.astype(np.float64))
  print("save : ", path+".npy")
  beauty_array(h ,path + ".txt")


path = "array/HP_bond_x"
h = P @ (-SxSx) @ np.linalg.inv(P)
if not os.path.isfile(path):
  np.save(path,h.astype(np.float64))
  print("save : ", path+".npy")
  beauty_array(h, path + ".txt")

path = "array/HP_bond_y"
h = P @ (-SySy) @ np.linalg.inv(P)
if not os.path.isfile(path):
  np.save(path,h.astype(np.float64))
  print("save : ", path+".npy")
  beauty_array(h,path + ".txt")

path = "array/HP_onsite"
h = P @ (m) @ np.linalg.inv(P)
if not os.path.isfile(path):
  np.save(path,h.astype(np.float64))
  print("save : ", path+".npy")
  beauty_array(h, path + ".txt")