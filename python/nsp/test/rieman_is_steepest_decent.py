import sys
sys.path.insert(0, "../")
import nsp
import numpy as np
import torch
import utils
import utils.optm as optm
import utils.lossfunc as lf
from importlib import reload

from nsp.optim import RiemanSGD
from matplotlib import pyplot as plt

x = np.random.randn(16,16)
x = (x+x.T)/2
x=torch.tensor(x)
loss_l1 = nsp.loss.L1(x, [4, 4])
loss_l2 = nsp.loss.L2(x, [4, 4])
loss_mes = nsp.loss.MES(x, [4, 4])

t = 0.001
model2 = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
sgd = RiemanSGD(model2, t)
loss_old = loss_l1(model2.matrix()).item()
loss_l1(model2.matrix()).backward()
S, U = sgd._riemannian_grad(model2._params)
W = model2.matrix().data
S = S/torch.linalg.norm(S@W)
loss_rieman_dir = loss_l1(torch.matrix_exp(-t*S)@W)

res = []
for _ in range(100000):
    S_ = np.random.randn(4,4)
    S_ = S_ - S_.T
    S_ = torch.Tensor(S_).to(torch.float64)
    S_ = S_/torch.linalg.norm(S_@W)
    loss_random = loss_l1(torch.matrix_exp(-t*S_)@W)
    res.append((loss_random-loss_rieman_dir).item())

fig, ax = plt.subplots()
ax.hist(res, bins=100)
ax.set_title('diff in the direction of rieman gradient, 1000 samples/lr = 0.001')
ax.set_xlabel('diff')
ax.set_ylabel('freq')
ax.legend()
fig.show()
plt.savefig('l1_rieman_is_steepest.jpg', dpi=400, bbox_inches="tight")



model2 = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
sgd = RiemanSGD(model2, t)
loss_old = loss_l2(model2.matrix()).item()
loss_l2(model2.matrix()).backward()
S, U = sgd._riemannian_grad(model2._params)
W = model2.matrix().data
S = S/torch.linalg.norm(S@W)
loss_rieman_dir = loss_l2(torch.matrix_exp(-t*S)@W)

res = []
for _ in range(100000):
    S_ = np.random.randn(4,4)
    S_ = S_ - S_.T
    S_ = torch.Tensor(S_).to(torch.float64)
    S_ = S_/torch.linalg.norm(S_@W)
    loss_random = loss_l2(torch.matrix_exp(-t*S_)@W)
    res.append((loss_random-loss_rieman_dir).item())

fig, ax = plt.subplots()
ax.hist(res, bins=100)
ax.set_title('diff in the direction of rieman gradient, 1000 samples/lr = 0.001')
ax.set_xlabel('diff')
ax.set_ylabel('freq')
ax.legend()
fig.show()
plt.savefig('l2_rieman_is_steepest.jpg', dpi=400, bbox_inches="tight")


# res = []
# for t in range(10000):
#     x = np.random.randn(16,16)
#     x = (x+x.T)/2
#     x=torch.tensor(x)
#     loss_l2 = nsp.loss.L2(x, [4, 4])
#     model = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
#     sgd = RiemanSGD(model, 0.001)
#     loss_old = loss_l2(model.matrix()).item()
#     loss_l2(model.matrix()).backward()
#     sgd.step()
#     loss_new = loss_l2(model.matrix()).item()
#     res.append((loss_new - loss_old))

# fig, ax = plt.subplots()
# ax.hist(res, bins=100)
# ax.set_title('diff in the direction of rieman gradient, 1000 samples/lr = 0.001')
# ax.set_xlabel('diff')
# ax.set_ylabel('freq')
# ax.legend()
# fig.show()
# plt.savefig('l2_rieman_grad_lr=0.001.jpg', dpi=400, bbox_inches="tight")
# # print(loss_new - loss_old)