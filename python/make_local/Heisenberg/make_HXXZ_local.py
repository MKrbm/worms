import numpy as np
import argparse
import sys
sys.path.append('../../reduce_nsp')
from nsp.utils.base_conv import *
from header import *
from nsp.utils.func import *
from nsp.utils.local2global import *
from nsp.utils.print import beauty_array
sys.path.insert(0, "..") 
from save_npy import *
import argparse
from datetime import datetime
from random import randint

# packages for multiprocessing
import psutil
import torch.multiprocessing as mp
import multiprocessing

lattice = [
    "1D",
]
u_algorithm = [
    "original",
]

loss = ["mes", "l1"]
J = [1, 1] #J, J_D
message = False
num_processes = psutil.cpu_count(logical=False)

parser = argparse.ArgumentParser(description='Optimize majumdar gosh')

# parameters defining models
parser.add_argument('-l','--lattice', help='lattice (model) Name', required=True, choices=lattice)
parser.add_argument('-u','--unitary_algorithm', help='algorithm determine local unitary matrix', default = "original", choices=u_algorithm)
parser.add_argument('-Jz','--coupling_z', help='coupling constant (Jz)', type = float, default = 1) # SxSx + SySy + 
parser.add_argument('-Jx','--coupling_x', help='coupling constant (Jx)', type = float, default = 1) 
parser.add_argument('-H','--magnetization', help='represent magnetization', type = float, default = 0) # + h * Sz

# optimizer settings
parser.add_argument('-loss','--loss', help='loss_methods', choices=loss, nargs='?', const='all',default="mes")
parser.add_argument('-M','--num_iter', help='# of iterations', type = int, default = 10)
parser.add_argument('-P','--num_processes', help='# of parallel process', type = int, default = num_processes)


args = parser.parse_args()
print(args)
Jz = args.coupling_z
Jx = args.coupling_x
h = args.magnetization
lat = args.lattice
ua = args.unitary_algorithm

Sz = np.zeros([2,2])
Sz[0,0] = 1/2
Sz[1,1] = -1/2
Sx = np.zeros([2,2])
Sx[1,0] = 1/2
Sx[0,1] = 1/2
Sy = np.zeros([2,2], dtype=np.complex64)
Sy[1,0] = 1j/2
Sy[0,1] = -1j/2

I = np.eye(2)

params_dict = dict(Jz=Jz, Jx=Jx, h=h)

a = ""
for k, v in params_dict.items():
    v = float(v)
    a += f"{k}_{v:.2}_"
params_str = a[:-1]


SzSz = np.kron(Sz,Sz).real.astype(np.float64)
SxSx = np.kron(Sx,Sx).real.astype(np.float64)
SySy = np.kron(Sy,Sy).real.astype(np.float64)
o = np.kron(I, Sz) + np.kron(Sz, I)

lh = Jz * SzSz + Jx*(SxSx + SySy)



if __name__ == "__main__":
    J_str = str(J).replace(" ", "")
    if lat == "1D":
        H = lh - h * o / 2
        H *= -1
        path = "array/1D"
        if ua == "original":
            path += "/original"
            path += "/" + params_str
            save_npy(path+"/H", [H])
            o /= 4 # divide by 4 because overwrapped 4 times.
            save_npy(path+"/Sz", [o])
