import sys
sys.path.append('..')
from nsp.solver import SymmSolver, UnitaryTransTs
from nsp.optim import RiemanUnitarySGD
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


def get_stat(res):
    tmp = []
    for i in range(len(res)-1):
        tmp += [res[i+1]/max(res[i], 0.001), res[i+1]]
    return tmp


lr = 0.0005
N = 1000
L = 10
res_comp_momentum = []
# for x in X_list:
model_rieman = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
LR = 0.1 * np.arange(1, L)
for _ in range(N):
    X = np.random.randn(4,4)
    X = X.T + X
    loss_l1 = nsp.loss.L1(torch.Tensor(X), [4])
    model_rieman.reset_params()    
    solver = UnitaryTransTs(RiemanUnitarySGD, copy.deepcopy(model_rieman), loss_l1, lr = lr, momentum=0)
    ret = solver.run(500, disable_message=True)
    tmp = []
    if (ret["fun"] < 0.01):
        continue
    tmp.append(ret["fun"])

    for t in LR:
        solver_momentum = UnitaryTransTs(RiemanUnitarySGD, copy.deepcopy(model_rieman), loss_l1, lr = lr, momentum=t)
        ret_momentum = solver_momentum.run(500, disable_message=True)
        if ret_momentum["fun"] < 0.001:
            break
        tmp.append(ret_momentum["fun"])
    tmp += [0] * (len(LR)+1 - len(tmp))
    res_comp_momentum.append(tmp)

pos = np.array([get_stat(res) for res in res_comp_momentum])
fig, ax = plt.subplots()
x1 = pos[:, 0]
y1 = pos[:, 1]
x2 = pos[:, 2]
y2 = pos[:, 3]



cmap = plt.get_cmap("autumn")
N = int(pos.shape[1]/2)-1
for i in range(int(pos.shape[1]/2)-1):
    x1 = pos[:, 2*i]
    y1 = pos[:, 2*i+1]
    x2 = pos[:, 2*i+2]
    y2 = pos[:, 2*i+3]
    C = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    ax.quiver(x1, y1, (x2-x1), (y2-y1), color = cmap(float(i)/N), angles='xy', scale_units='xy', scale=1, headwidth=4, width=0.003)


cmap = plt.get_cmap("winter")
N = int(pos.shape[1]/2)
for i in range(int(pos.shape[1]/2)):
    x1 = pos[:, 2*i]
    y1 = pos[:, 2*i+1]
    ax.scatter(x = x1, y = y1, s = 5, label = "momentum={}".format(LR[i]),
    color = cmap(float(i)/N))
ax.set_ylabel("loss")
ax.set_xlabel("loss rate (loss_{\mu=t} / loss_{\mu=t-1})")
# fig.show()e
plt.savefig('images/methods_compare.jpg', dpi=400, bbox_inches="tight")


# print(ret_rieman)