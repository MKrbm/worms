from header import *

lr = 0.001
D = 16

import copy

model_rf = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model_rf2 = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)


res = []
# X_list = np.load("X_list.npy")
m = 5
# for X in X_list[:3]:
sparse = 1/D
for _ in range(1000):
    X = np.random.randn(D**2, D**2)
    X = (X + X.T)/2
    mask = X>=(1/(D**2))
    X[mask] = 0
    loss_orth = nsp.loss.MES(torch.Tensor(X), [D,D], inv = model_rf._inv) 
    tmp = []


    best_ans = 1E10
    for _ in range(m):
        model_rf.reset_params()
        solver = UnitaryTransTs(RiemanUnitarySGD, model_rf, loss_orth, lr = lr, momentum=0.1, prt = False)
        ret = solver.run(3000, disable_message=False)
        best_ans = min(ret.fun, best_ans)
    tmp.append(best_ans)

    best_ans = 1E10
    for _ in range(m):
        model_rf2.reset_params()
        solver = UnitaryTransTs(RiemanUnitaryCG, model_rf, loss_orth, lr = lr, momentum=0.1, prt = False)
        ret = solver.run(500, disable_message=False)
        best_ans = min(ret.fun, best_ans)
    tmp.append(best_ans)

    res.append(tmp)
    print(tmp)
res = np.array(res)

fig, ax = plt.subplots()
ax.hist(res[:,1] - res[:, 0], label="samples", bins=100)
ax.legend()


ax.set_title('compare riemannian float unitary with float SL')
ax.set_xlabel('SL - O')
ax.set_ylabel('freq')
save_fig(plt,"images",f'SL_vs_orth_sparse_D={D}_m={m}_L={type(loss_orth).__name__}.jpeg', dpi=400)