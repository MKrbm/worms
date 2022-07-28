from header import *

D = 10
set_seed(3342432)
X = np.random.randn(D**2,D**2)
X = (X + X.T)/2
E, V = np.linalg.eigh(X)
P = np.diag(np.sign(V[:,-1]))
model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model.reset_params()
model2 = copy.deepcopy(model)
loss = nsp.loss.MES(torch.Tensor(X), [D,D], inv = model._inv)
loss2 = nsp.loss.MES(torch.Tensor(P@X@P), [D,D], inv = model._inv)
# print(model.matrix(), model2.matrix())
solver1 = UnitarySymmTs(RiemanUnitaryCG, model, loss, lr = 0.1)
ret_cg = solver1.run(100, disable_message=False)

solver2 = UnitarySymmTs(RiemanUnitaryCG, model2, loss2, lr = 0.1)
ret_cg2 = solver2.run(100, disable_message=False)

