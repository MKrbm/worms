# optimize random matrix with negative-sign by unitary transformation

import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import sparse
import scipy
import sys
sys.path.append('..')
from importlib import reload
from utils import optm
from utils.functions import *
import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


N_iter = 3000
L = 4
torch.manual_seed(0)
X = torch.rand((L**2,L**2),dtype=torch.float64)*2-1
X = X + X.T

# with no_grad:
E, V = np.linalg.eigh(np.array(X))


# chose model and optimizer here. ather than unitary solver don't work yet.
# model = optm.unitary_solver([L,L],syms=False)
model = optm.unitary_solver([L,L],syms=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for t in range(N_iter):
    y = model(X)
    loss = optm.loss_1(y)
    if t % 1000 == 0:
#         print(y)
        print("iteration : {:4d}   loss : {:.3f}".format(t,loss.item()))
#         print(next(model_diag.parameters()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



E2, V2 = np.linalg.eigh(absolute_map(np.array(X.data)))
E3, V3 = np.linalg.eigh(absolute_map(np.array(y.data)))


print("original ge : ", E[0])
print("absolute ge before optm: ", E2[0])
print("absolute ge after optm: ", E3[0])

