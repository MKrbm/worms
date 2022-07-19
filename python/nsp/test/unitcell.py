from header import *

D = 10




set_seed(3342432)
X = np.random.randn(D**2,D**2)
X = X.reshape(D,D,D,D)
# X = X + X.transpose(1,0,3,2)
X = X.reshape(D**2,D**2)
X = (X + X.T)/2
# model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
# model.reset_params()
# X1 = model.matrix().data.numpy() @ np.diag(np.random.rand(10)) @ model.matrix().data.numpy().T
# model.reset_params()
# X2 = model.matrix().data.numpy() @ np.diag(np.random.rand(10)) @ model.matrix().data.numpy().T
# X = np.kron(X1, X2)
# X = np.random.randn(D**2,D**2)
# X = X.reshape(D,D,D,D)
# X = X + X.transpose(1,0,3,2)
# X = X.reshape(D**2,D**2)
# X = (X + X.T)/2

# X = np.random.randn(D**2, D**2)
# X = (X + X.T)/2
# mask = np.abs(X)>=(1/(D))
# X[mask] = 0

models = [nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64) for _ in range(5)]
model = copy.deepcopy(models[0])
loss = nsp.loss.MES(torch.Tensor(X), [D,D], inv = model._inv)
loss1 = nsp.loss.MES(torch.Tensor(X/4), [D,D], inv = model._inv)
cg1 = RiemanUnitaryCG(model, loss, lr = 0.1)
cg2 = RiemanUnitaryCG2([(models[i], models[i+1]) for i in range(4)], [loss1]*4)

solver1 = UnitarySymmTs(RiemanUnitaryCG, model, loss, lr = 0.1)
solver2 = UnitaryLocalTs(cg2)

# ret_cg = solver2.run(1000, disable_message=False)
ret_cg = solver1.run(300, disable_message=False)
