import sys
sys.path.insert(0, "../")
import nsp
import numpy as np
import torch
import utils
import utils.optm as optm
import utils.lossfunc as lf
from importlib import reload
model = nsp.model.UnitaryRiemanGenerator(4, dtype=np.float64)
M = model._get_matrix()

model2 = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.float64)
M2 = model2._get_matrix()


