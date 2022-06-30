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
loss_l1 = nsp.loss.l1(x, [4, 4])
loss_l2 = nsp.loss.l2(x, [4, 4])
loss_mes = nsp.loss.mes(x, [4, 4])

res = []
for t in range(10000):
    x = np.random.randn(16,16)
    x = (x+x.T)/2
    x=torch.tensor(x)
    loss_l1 = nsp.loss.l1(x, [4, 4])
    model = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
    sgd = RiemanSGD(model, 0.001)
    loss_old = loss_l1(model.matrix()).item()
    loss_l1(model.matrix()).backward()
    sgd.step()
    loss_new = loss_l1(model.matrix()).item()
    res.append((loss_new - loss_old))

fig, ax = plt.subplots()
ax.hist(res, bins=100)
ax.set_title('diff in the direction of rieman gradient, 1000 samples/lr = 0.001')
ax.set_xlabel('diff')
ax.set_ylabel('freq')
ax.legend()
fig.show()
plt.savefig('l1_rieman_grad_lr=0.001.jpg', dpi=400, bbox_inches="tight")


res = []
for t in range(10000):
    x = np.random.randn(16,16)
    x = (x+x.T)/2
    x=torch.tensor(x)
    loss_l2 = nsp.loss.l2(x, [4, 4])
    model = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
    sgd = RiemanSGD(model, 0.001)
    loss_old = loss_l2(model.matrix()).item()
    loss_l2(model.matrix()).backward()
    sgd.step()
    loss_new = loss_l2(model.matrix()).item()
    res.append((loss_new - loss_old))

fig, ax = plt.subplots()
ax.hist(res, bins=100)
ax.set_title('diff in the direction of rieman gradient, 1000 samples/lr = 0.001')
ax.set_xlabel('diff')
ax.set_ylabel('freq')
ax.legend()
fig.show()
plt.savefig('l2_rieman_grad_lr=0.001.jpg', dpi=400, bbox_inches="tight")
# print(loss_new - loss_old)