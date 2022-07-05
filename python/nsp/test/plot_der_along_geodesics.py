import sys
sys.path.insert(0, "../")
import nsp
import numpy as np
import torch
import utils
import utils.optm as optm
import utils.lossfunc as lf
from importlib import reload

from nsp.optim import RiemanSGD, RiemanCG
from matplotlib import pyplot as plt

X = np.random.randn(4, 4)
X = X.T + X
loss_l1 = nsp.loss.L1(X, [4])
loss_l2 = nsp.loss.L2(X, [4])
loss_mes = nsp.loss.MES(X, [4])
t = 0.001
ret_min_grad = 1e10
model = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
W = model.matrix().data
L1=loss_l1(model.matrix())
L1.backward()
sgd = RiemanCG(model, loss_l1, t)
S, U = sgd._riemannian_grad(model._params)

H = S.data
t =  torch.tensor([0], requires_grad=True, dtype=torch.float64)
res = []
for t_ in np.arange(-300, 300, 1)*0.001:
    t.data[0] = t_
    U = torch.matrix_exp(-t*H)@W
    loss = loss_l1(U)
    g = torch.autograd.grad(loss, t, create_graph=True)
    res.append(g[0].item())

plt.figure(figsize=(10,5))
plt.plot(np.arange(-300, 300, 1)*0.001, res)
plt.title("plot derivative along geodesics in the direction of riemannian gradient from random unitary")
plt.xlabel("t")
plt.ylabel("derivative w.r.t t")
plt.savefig("images/der_along_geodesics.jpeg", dpi=800)