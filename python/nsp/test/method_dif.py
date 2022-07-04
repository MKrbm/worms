import sys
sys.path.append('..')
from nsp.solver import SymmSolver, UnitarySymmTs
from nsp.optim import RiemanSGD
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


model_lie = nsp.model.UnitaryGenerator(4, dtype=torch.float64)
model_rieman = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
model_lie.reset_params()
model_adam = copy.deepcopy(model_lie)


X = np.random.randn(4,4)
X = X.T + X
loss_l1 = nsp.loss.L1(torch.Tensor(X), [4])

lr = 0.001
# solver_rieman = UnitarySymmTs(RiemanSGD, model_rieman, loss_l1, lr = lr, momentum=0.01)
# ret_rieman = solver_rieman.run(500, disable_message=True)

solver_lie_sgd = UnitarySymmTs(torch.optim.SGD, model_lie, loss_l1, lr = lr, momentum=0.01)
ret_lie_sgd = solver_lie_sgd.run(500, disable_message=True)



# print(ret_rieman)