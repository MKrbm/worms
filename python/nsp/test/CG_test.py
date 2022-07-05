import sys
sys.path.append('..')
from nsp.solver import SymmSolver, UnitarySymmTs
from nsp.optim import RiemanSGD, RiemanCG
import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import sparse
import scipy
from utils import lossfunc as lf

import sys
import utils
import utils.optm as optm
import utils.lossfunc as lf
import numpy as np
import torch
from importlib import reload
import nsp
import copy
from matplotlib import pyplot as plt
import random
from nsp.utils.func import *


set_seed(33244233)
N = 14
X = np.random.randn(N, N)
X = X.T + X
loss_l1 = nsp.loss.MES(X, [N])
t = 0.001
ret_min_grad = 1e10
model = nsp.model.UnitaryRiemanGenerator(N, dtype=torch.float64)
solver = UnitarySymmTs(RiemanCG, model, loss_l1, lr = 0.005, momentum=0.1)

# model = nsp.model.UnitaryGenerator(N, dtype=torch.float64)
# solver = UnitarySymmTs(torch.optim.SGD, model, loss_l1, lr = 0.001, momentum=0)

# solver.run(3000)

# print("start rieman sgd")
# solver = UnitarySymmTs(RiemanSGD, model, loss_l1, lr = 0.001)
solver.run(1000)

print("start plot figures")
cg = RiemanCG(model, loss_l1, 0.001)
S, W = cg._riemannian_grad(model._params)
H = S.data
res_grad = []
res = []
for t_ in np.arange(-300, 300, 1)*0.001:
    t =  torch.tensor([t_], requires_grad=True, dtype=torch.float64)
    U = torch.matrix_exp(-t*H)@W
    loss = loss_l1(U)
    g = torch.autograd.grad(loss, t, create_graph=True)
    res.append(loss.item())
    res_grad.append(g[0].item())    



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_title('measure along geodesics in direction of riemannian gradient')
ax1.set_xlabel('t along geodesics')
ax1.set_ylabel('loss', color='g')
ax2.set_ylabel('loss_grad', color='b')
ax1.plot(np.arange(-300, 300, 1)*0.001, res, color="g")
ax2.plot(np.arange(-300, 300, 1)*0.001, res_grad, color="b")
fig.show()
plt.savefig("images/cg_along_geodesics_at_localminimum.jpeg", dpi=400)



S_ = np.random.randn(14,14)
S_ = S_ - S_.T
S_ = torch.Tensor(S_).to(torch.float64)
S = S_/torch.linalg.norm(S_@W)
H = S.data
res_grad = []
res = []
for t_ in np.arange(-300, 300, 1)*0.001:
    t =  torch.tensor([t_], requires_grad=True, dtype=torch.float64)
    U = torch.matrix_exp(-t*H)@W
    loss = loss_l1(U)
    g = torch.autograd.grad(loss, t, create_graph=True)
    res.append(loss.item())
    res_grad.append(g[0].item())    

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_title('measure along geodesics in direction of random skew matrix')
ax1.set_xlabel('t along geodesics')
ax1.set_ylabel('loss', color='g')
ax2.set_ylabel('loss_grad', color='b')
ax1.plot(np.arange(-300, 300, 1)*0.001, res, color="g")
ax2.plot(np.arange(-300, 300, 1)*0.001, res_grad, color="b")
fig.show()
plt.savefig("images/cg_along_geodesics_at_localminimum2.jpeg", dpi=400)