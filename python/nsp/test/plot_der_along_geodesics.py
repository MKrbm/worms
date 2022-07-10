from header import *

D = 2
set_seed(3080984)
X = np.random.randn(D**2,D**2)
X = X.T + X
loss = nsp.loss.L2(X, [D,D])
# loss_l2 = nsp.loss.L2(X, [D,D])
# loss_mes = nsp.loss.MES(X, [D,D])
t = 0.001
ret_min_grad = 1e10
model = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.complex128)
model.reset_params()
solver = UnitarySymmTs(RiemanUnitarySGD, model, loss, lr = t, momentum=0.1, pout = True)
ret = solver.run(10, disable_message=False)

model = ret.model
W = model.matrix().data
L1=loss(model.matrix())
L1.backward()
sgd = RiemanUnitaryCG(model, loss, t)
S, U = sgd._riemannian_grad(model._params)

H = S.data
t =  torch.tensor([0], requires_grad=True, dtype=torch.float64)
res = []
for t_ in np.arange(-300, 300, 1)*0.001:
    t.data[0] = t_
    U = torch.matrix_exp(-t*H)@W
    loss_ = loss(U)
    g = torch.autograd.grad(loss_, t, create_graph=True)
    res.append(g[0].item())

plt.figure(figsize=(10,5))
plt.plot(np.arange(-300, 300, 1)*0.001, res)
plt.title("plot derivative along geodesics in the direction of riemannian gradient from random unitary")
plt.xlabel("t")
plt.ylabel("derivative w.r.t t")
save_fig(plt,"images",f"der_along_geodesics_complex_{type(loss).__name__}.jpeg", dpi=800)