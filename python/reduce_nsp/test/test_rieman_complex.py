from header import *

lr = 0.005
D = 4
import copy

# model_f = nsp.model.UnitaryGenerator(D, dtype=torch.float64)
# model_g = nsp.model.UnitaryGenerator(D, dtype=torch.complex128)
# model_rf = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model_rg = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)

set_seed(33333)

# X = np.random.randn(D**2,D**2)
X = np.random.randn(D,D)

X = X.T + X
loss = nsp.loss.L1(torch.Tensor(X), [D])
sgd = RiemanUnitarySGD(model_rg, 0.01)
solver = UnitaryTransTs(RiemanUnitarySGD, model_rg, loss, lr = lr, momentum=0.1, prt = True)
loss(model_rg.matrix()).backward()
H, _ = sgd._riemannian_grad(model_rg._params)
# print