import numpy as np
from scipy import sparse
import os
from functions import *



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

h1 = -(sparse.kron(sparse.kron(I,Sz,format='csr'), sparse.kron(Sz,I,format='csr'),format='csr') 
       - sparse.kron(sparse.kron(I,Sx,format='csr'), sparse.kron(Sx,I,format='csr'),format='csr')
       - sparse.kron(sparse.kron(I,Sy,format='csr'), sparse.kron(Sy,I,format='csr'),format='csr') 
     ).real


h2 = -(sparse.kron(sparse.kron(Sz,I,format='csr'), sparse.kron(Sz,I,format='csr'),format='csr') 
       - sparse.kron(sparse.kron(Sx,I,format='csr'), sparse.kron(Sx,I,format='csr'),format='csr')
       - sparse.kron(sparse.kron(Sy,I,format='csr'), sparse.kron(Sy,I,format='csr'),format='csr') 
     ).real


h3 = -(sparse.kron(sparse.kron(Sz,Sz,format='csr'), sparse.kron(I,I,format='csr'),format='csr') 
       + sparse.kron(sparse.kron(Sx,Sx,format='csr'), sparse.kron(I,I,format='csr'),format='csr')
       + sparse.kron(sparse.kron(Sy,Sy,format='csr'), sparse.kron(I,I,format='csr'),format='csr') 
     ).real

h4 = -(sparse.kron(sparse.kron(I,I,format='csr'), sparse.kron(Sz,Sz,format='csr'),format='csr') 
       + sparse.kron(sparse.kron(I,I,format='csr'), sparse.kron(Sx,Sx,format='csr'),format='csr')
       + sparse.kron(sparse.kron(I,I,format='csr'), sparse.kron(Sy,Sy,format='csr'),format='csr') 
     ).real




h1_ = -(sparse.kron(sparse.kron(Sz,I,format='csr'), sparse.kron(I,Sz,format='csr'),format='csr') 
       - sparse.kron(sparse.kron(Sx,I,format='csr'), sparse.kron(I,Sx,format='csr'),format='csr')
       - sparse.kron(sparse.kron(Sy,I,format='csr'), sparse.kron(I,Sy,format='csr'),format='csr') 
     ).real

h2_ = -(sparse.kron(sparse.kron(I,Sz,format='csr'), sparse.kron(I,Sz,format='csr'),format='csr') 
       - sparse.kron(sparse.kron(I,Sx,format='csr'), sparse.kron(I,Sx,format='csr'),format='csr')
       - sparse.kron(sparse.kron(I,Sy,format='csr'), sparse.kron(I,Sy,format='csr'),format='csr') 
     ).real



h = h1 + h2

h_ = h1_ + h2_

on_site = h3/4 + h4/4

u = np.array([
    [0,1,0,0],
    [1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [-1/np.sqrt(2), 0, 1/np.sqrt(2), 0],
    [0,0,0,1]
])

print(u)
u = sparse.csr_matrix(u)
U = sparse.kron(u, u,format='csr')

H = U.T @ h @ U
H2 = U.T @ h_ @ U
ON = U.T @ on_site @ U

# a = -0.5
# H[state2num([0,0]), state2num([0,1])] = a
# H[state2num([0,1]), state2num([0,0])] = a
# H[state2num([0,0]), state2num([1,0])] = a
# H[state2num([1,0]), state2num([0,0])] = a


# H2[state2num([0,0]), state2num([0,1])] = a
# H2[state2num([0,1]), state2num([0,0])] = a
# H2[state2num([0,0]), state2num([1,0])] = a
# H2[state2num([1,0]), state2num([0,0])] = a

index = sparse.find(H+ON)
print("type 1 bond operator\n")
for i,j, ele in zip(index[0], index[1], index[2]):
    print(num2state(i, 2), num2state(j, 2), "\t {:.3f}".format(ele))


print("--"*10)
print("type 2 bond operator\n")
index = sparse.find(H2+ON)
for i,j, ele in zip(index[0], index[1], index[2]):
  print(num2state(i, 2), num2state(j, 2), "\t {:.3f}".format(ele))





path1 = "array/SS_bond1"
if not os.path.isfile(path1):
  np.save(path1,H.toarray())
  print(H[0,1])
  print("save : ", path1+".npy")
  beauty_array(H,path1 + ".txt")


path2 = "array/SS_bond2"
if not os.path.isfile(path2):
  np.save(path2,H2.toarray())
  print("save : ", path2+".npy")
  beauty_array(H2,path2 + ".txt")

path3 = "array/SS_onsite"
if not os.path.isfile(path3):
  np.save(path3,ON.toarray())
  print("save : ", path3+".npy")
  beauty_array(ON,path3 + ".txt")


beauty_array(ON + H2, "array/local_ham.txt")



