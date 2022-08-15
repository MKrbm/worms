import numpy as np
from scipy import sparse
import os
import sys
sys.path.insert(0, "../nsp") 
from nsp.utils.func import *
from nsp.utils.print import beauty_array



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

path = "../array/majumdar_ghosh/original/0"
if not os.path.isfile(path):
  np.save(path,lh)
  print("save : ", path+".npy")
  beauty_array(lh,path + ".txt")

path = "../array/majumdar_ghosh/original/1"
if not os.path.isfile(path):
  np.save(path,lh/2)
  print("save : ", path+".npy")
  beauty_array(lh/2,path + ".txt")




