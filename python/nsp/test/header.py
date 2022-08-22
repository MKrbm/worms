import sys
sys.path.append('..')
from nsp.solver import SymmSolver, UnitaryTransTs, UnitaryNonTransTs
from nsp.optim import *
import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import sparse
from scipy.optimize import OptimizeResult
import scipy

import sys
import numpy as np
import torch
from importlib import reload
import nsp
import copy
from matplotlib import pyplot as plt
import random
from nsp.utils.func import *
from nsp.utils import save_fig