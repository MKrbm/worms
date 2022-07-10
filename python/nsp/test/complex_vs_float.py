from header import *

lr = 0.001
D = 3

import copy

model_f = nsp.model.UnitaryGenerator(D, dtype=torch.float64)
model_g = nsp.model.UnitaryGenerator(D, dtype=torch.complex128)
model_rf = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model_rg = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.complex128)


res = []
# X_list = np.load("X_list.npy")
m = 5
# for X in X_list[:3]:
sparse = 1/D
for _ in range(100):
    X = np.random.randn(D**2, D**2)
    X = (X + X.T)/2
    mask = X>=(1/D)
    X[mask] = 0
    loss = nsp.loss.MES(torch.Tensor(X), [D,D])    
    tmp = []

    best_ans = 1E10
    for _ in range(m):
        model_g.reset_params()
        solver = UnitarySymmTs(torch.optim.SGD, model_g, loss, lr = lr, momentum=0.1, pout = False)
        ret = solver.run(500, disable_message=True)
        best_ans = min(ret.fun, best_ans)
    tmp.append(best_ans)

    best_ans = 1E10
    for _ in range(m):
        model_rf.reset_params()
        solver = UnitarySymmTs(RiemanUnitaryCG, model_rf, loss, lr = lr, momentum=0.1, pout = False)
        ret = solver.run(500, disable_message=True)
        best_ans = min(ret.fun, best_ans)
    tmp.append(best_ans)

    best_ans = 1E10
    for _ in range(m):
        model_rg.reset_params()
        solver = UnitarySymmTs(RiemanUnitaryCG, model_rg, loss, lr = lr, momentum=0.1, pout = False)
        ret = solver.run(500, disable_message=True)
        best_ans = min(ret.fun, best_ans)
    tmp.append(best_ans)

    res.append(tmp)
    print(tmp)
res = np.array(res)

fig, ax = plt.subplots()
x = res[:,1] - res[:, 0]
y = res[:,1] - res[:, 2]
ax.scatter(x, y, s=20, label="samples")
ax.legend()


ax.set_title('compare riemannian float with torch sgd complex and riemannain complex')
ax.set_xlabel('float - torch complex')
ax.set_ylabel('float - riemannian complex')
save_fig(plt,"images",f'complex_vs_float_sparse_D={D}_m={m}_L={type(loss).__name__}.jpeg', dpi=400)