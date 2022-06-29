import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import sparse
import scipy

import sys
sys.path.append('..')
import numpy as np
import torch
from importlib import reload
import os
print(os.getcwd())
import nsp


L = 4
np.random.seed(4)
X = np.random.rand(L*L).reshape(L, L)-0.5
X = X + X.T + np.eye(L)

from nsp.solver import sym_solver


model = nsp.model.unitary_generator(2, dtype=torch.float64, spherical=False)
loss = nsp.loss.mes(torch.tensor(X), [2,2])

# print(model._params.data)
model._params.data[:] = 0

solver = nsp.solver.unitary_symm_ts(nsp.optim.sign, model, loss, lr = 0.005, decay_rate = 0.999)
# solver = nsp.solver.unitary_symm_ts(torch.optim.Adam, model, loss, lr = 0.001)

solver.run(1000)
print("\n",model._params.data)
print("\n\n")
