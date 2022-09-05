import numpy as np
from scipy import sparse
import os
import sys
sys.path.insert(0, "../nsp") 
from nsp.utils.func import *
from nsp.utils.print import beauty_array
import argparse

lattice = [
    "original",
    "chain8",
    "3site",
    "chain8_af"
]

parser = argparse.ArgumentParser(description='Reproduce original paper results')
parser = argparse.ArgumentParser(description='Reproduce original paper results')
parser.add_argument('-l','--lattice', help='lattice (model) Name', required=True, choices=lattice)
args = vars(parser.parse_args())
lat = args["lattice"]

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

lh = -lh # use minus of local hamiltonian for monte-carlo (exp(-beta H ))

if lat == "original":
    H = lh
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
elif lat == "chain8":
    print("1D chain lattice")
    bonds = [[0,1], [0, 2], [1, 2]]
    lh2 = sum_ham(lh/2, bonds, 3, 2)
    LH = sum_ham(lh2/2, [[0,1,2], [3, 4, 5]], 6, 2) + sum_ham(lh2, [[1, 2, 3], [2, 3, 4]], 6, 2)
    # LH = sum_ham(LH/2, [[0, 1], [2, 3]], 4, 8) + sum_ham(LH, [[1, 2]], 4, 8)
    lh = LH
    path = "../array/majumdar_ghosh/1D_chain8/"
    name = "0"
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isfile(path):
        np.save(path + name,lh)
        print("save : ", path+name + ".npy")
        # beauty_array(lh,path + name + ".txt")

elif lat == "3site":
    print("3site operator lattice")
    bonds = [[0,1], [0, 2], [1, 2]]
    lh2 = sum_ham(lh/2, bonds, 3, 2)
    path = "../array/majumdar_ghosh/3site/"
    if not os.path.exists(path):
        os.makedirs(path)

    name = "0"
    if not os.path.isfile(path):
        np.save(path + name,lh2)
        print("save : ", path+name + ".npy")
        # beauty_array(lh,path + name + ".txt")

elif lat == "chain8_af":
    print("1D chain lattice")
    bonds = [[0,1], [0, 2], [1, 2]]
    lh2 = sum_ham(lh/2, bonds, 3, 2)
    LH = sum_ham(lh2/2, [[0,1,2], [3, 4, 5]], 6, 2) + sum_ham(lh2, [[1, 2, 3], [2, 3, 4]], 6, 2)
    act = [8, 8]
    I_not1 = np.logical_not(np.eye(act[0]))
    I_not2 = np.logical_not(np.eye(act[1]))
    ML = np.kron(I_not1, np.eye(act[1]))
    MR = np.kron(np.eye(act[0]), I_not2)
    MI = np.kron(I_not1, I_not2) + np.eye(np.prod(act))
    LH_3 = (np.kron(np.eye(act[0]), ML * LH) + np.kron(MR * LH,np.eye(act[1])))
    LH_2 = MI * LH
    path = "../array/majumdar_ghosh/1D_chain8_af/"
    names = ["0", "1"]
    hams = [LH_2, LH_3]
    if not os.path.exists(path):
        os.makedirs(path)
    
    for name, ham in zip(names, hams):
        if not os.path.isfile(path):
            np.save(path + name, ham)
            print(ham.shape)
            print("save : ", path+name + ".npy")





