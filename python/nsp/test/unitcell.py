from header import *

D = 10
set_seed(3342432)

model1 = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model2 = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model3 = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model4 = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model = copy.deepcopy(model1)

X1 = model4.matrix().data.numpy() @ np.diag(np.random.rand(10)) @ model4.matrix().data.numpy().T
X2 = model3.matrix().data.numpy() @ np.diag(np.random.rand(10)) @ model3.matrix().data.numpy().T
X = np.kron(X1, X2)



# X = np.random.randn(D**2, D**2)
# X = (X + X.T)/2
# mask = np.abs(X)>=(1/(D))
# X[mask] = 0

loss = nsp.loss.MES(torch.Tensor(X), [D,D], inv = model1._inv)
cg1 = RiemanUnitaryCG(model, loss, lr = 0.1)
cg2 = RiemanUnitaryCG2([(model1, model2)], [loss])

solver1 = UnitarySymmTs(RiemanUnitaryCG, model, loss, lr = 0.1)
solver2 = UnitaryLocalTs(cg2)

ret_cg = solver1.run(1000, disable_message=False)

ret_cg = solver2.run(1000, disable_message=False)