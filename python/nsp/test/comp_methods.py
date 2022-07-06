from header import *

lr = 0.005
D = 10
import copy

model_lie = nsp.model.UnitaryGenerator(D, dtype=torch.float64)
model_rieman = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
set_seed(20220705)
res = []
X_list = []
for _ in range(100):
    X = np.random.randn(D**2,D**2)
    X = X.T + X
    loss = nsp.loss.L1(torch.Tensor(X), [D,D])


    model_lie.reset_params()
    model_adam = copy.deepcopy(model_lie)
    model_rieman.set_params(model_lie.matrix().data.view(-1))
    model_cg = copy.deepcopy(model_rieman)



    solver_cg = UnitarySymmTs(RiemanCG, model_cg, loss, lr = lr)
    ret_cg = solver_cg.run(500, disable_message=True)

    solver_rieman = UnitarySymmTs(RiemanSGD, model_rieman, loss, lr = lr, momentum=0.0)
    ret_rieman = solver_rieman.run(500, disable_message=True)

    solver_lie_sgd = UnitarySymmTs(torch.optim.SGD, model_lie, loss, lr = lr)
    ret_lie_sgd = solver_lie_sgd.run(500, disable_message=True)

    solver_lie_adam = UnitarySymmTs(torch.optim.Adam, model_adam, loss, lr = lr)
    ret_lie_adam = solver_lie_adam.run(500, disable_message=True)
    
    res.append([ret_lie_sgd["fun"],  ret_lie_adam["fun"], ret_rieman["fun"], ret_cg["fun"]])
    if  ret_rieman["fun"] > 0.1:
        X_list.append(X)

from matplotlib import pyplot as plt
fig, ax = plt.subplots()
res_rie = np.array(res)[:,2]
x = np.array(res)[:,0] - res_rie
y = np.array(res)[:,1] - res_rie
ax.scatter(x, y, s=20, label="samples")
ax.legend()


ax.set_title('compare methods to rieman with momentum (share same initial U)')
ax.set_xlabel('sgd - rieman')
ax.set_ylabel('adam - rieman')
plt.savefig(f'images/comp_meths_resdiff_sgd_D={D}_L={type(loss).__name__}.jpeg', dpi=400)

fig, ax = plt.subplots()
res_rie = np.array(res)[:,3]
x = np.array(res)[:,0] - res_rie
y = np.array(res)[:,1] - res_rie
ax.scatter(x, y, s=20, label="samples")
ax.legend()


ax.set_title('compare methods to rieman cg with momentum (share same initial U)')
ax.set_xlabel('sgd - rieman')
ax.set_ylabel('adam - rieman')
plt.savefig(f'images/comp_meths_resdiff_cg_D={D}_L={type(loss).__name__}.jpeg', dpi=400)


fig, ax = plt.subplots()
res = np.array(res)
ax.hist(res[:,2] - res[:,3], label="samples", bins=100)
ax.legend()


ax.set_title('compare CG vs SGD (2D)')
ax.set_xlabel('loss_sgd - loss_cg')
ax.set_ylabel("frequency")
plt.savefig(f'images/hist_sgd_vs_cg_D={D}_L={type(loss).__name__}.jpeg', dpi=400)
