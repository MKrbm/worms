from header import *

lr = 0.001
D = 2

model = nsp.model.SlRiemanGenerator(D, dtype=torch.float64)
model.reset_params(unitary=True)
set_seed(3342432)

X = np.random.randn(D**2,D**2)
X = X.T + X 
loss = nsp.loss.MES_SL(torch.Tensor(X), [D,D], inv = model._inv)

model.reset_params()
loss(model.matrix()).backward()
slcg = RiemanSlCG(model, loss)
S, W = slcg._riemannian_grad(model._params)
solver_cg = UnitarySymmTs(RiemanSlCG, model, loss)
ret_cg = solver_cg.run(500, disable_message=False)