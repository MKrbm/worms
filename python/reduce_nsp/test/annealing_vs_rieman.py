from header import *

set_seed(20220705)

overwrite = False
lr = 0.005
D = 2
model_sgd_0 = nsp.model.UnitaryRiemanGenerator(D, dtype=torch.float64)
model_sgd_1 = copy.deepcopy(model_sgd_0)
model_sgd_2 = copy.deepcopy(model_sgd_0)
model_cg = copy.deepcopy(model_sgd_0)

model_lie = nsp.model.UnitaryGenerator(D, dtype=torch.float64)
bounds = [[-30, 30]] * model_lie._n_params

res = []
for _ in range(100):
    tmp = []
    X = np.random.randn(D**2,D**2)
    X = X.T + X
    loss = nsp.loss.L1(torch.Tensor(X), [D,D])
    solver = UnitaryTransTs(RiemanUnitarySGD, model_sgd_0, loss, lr = lr, momentum=0, prt = False)
    ret = solver.run(500, disable_message=True)
    tmp.append(ret["fun"])
    
    solver = UnitaryTransTs(RiemanUnitarySGD, model_sgd_1, loss, lr = lr, momentum=0.1, prt = False)
    ret = solver.run(500, disable_message=True)
    tmp.append(ret["fun"])

    solver = UnitaryTransTs(RiemanUnitarySGD, model_sgd_2, loss, lr = lr, momentum=0.5, prt = False)
    ret = solver.run(500, disable_message=True)
    tmp.append(ret["fun"])
    
    solver = UnitaryTransTs(RiemanUnitaryCG, model_cg, loss, lr = lr, momentum=0, prt = False)
    ret = solver.run(500, disable_message=True)
    tmp.append(ret["fun"])
    
    ret = scipy.optimize.dual_annealing(SymmSolver(model_lie, loss, False), bounds = bounds, restart_temp_ratio = 1e-3,
                                  visit = 2.7, initial_temp = 10**4, maxiter = 200)
    tmp.append(ret.fun)
    res.append(tmp)

res = np.array(res)
fig, ax = plt.subplots()
res = np.array(res)
ax.hist(res[:,0] - res[:,-1], label="samples", bins=100)
ax.legend()


ax.set_title('annealing vs SGD (momentum = 0)')
ax.set_xlabel('loss_aneeling - loss_sgd')
ax.set_ylabel("frequency")
save_fig(plt, "images", f'hist_annealing_vs_riemansgd_D={D}_L={type(loss).__name__}.jpeg', 400, overwrite = overwrite)


res = np.array(res)
fig, ax = plt.subplots()
res = np.array(res)
ax.hist(res[:,1] - res[:,-1], label="samples", bins=100)
ax.legend()


ax.set_title('annealing vs SGD (momentum = 0.1)')
ax.set_xlabel('loss_aneeling - loss_sgd')
ax.set_ylabel("frequency")
save_fig(plt,"images", f'hist_annealing_vs_riemansgd1_D={D}_L={type(loss).__name__}.jpeg', 400, overwrite = overwrite)


res = np.array(res)
fig, ax = plt.subplots()
res = np.array(res)
ax.hist(res[:,2] - res[:,-1], label="samples", bins=100)
ax.legend()


ax.set_title('annealing vs SGD (momentum = 0.5)')
ax.set_xlabel('loss_aneeling - loss_sgd')
ax.set_ylabel("frequency")
save_fig(plt,"images" ,f'hist_annealing_vs_riemansgd2_D={D}_L={type(loss).__name__}.jpeg', 400, overwrite = overwrite)


res = np.array(res)
fig, ax = plt.subplots()
res = np.array(res)
ax.hist(res[:,3] - res[:,-1], label="samples", bins=100)
ax.legend()


ax.set_title('annealing vs CG')
ax.set_xlabel('loss_aneeling - loss_cg')
ax.set_ylabel("frequency")
save_fig(plt,"images", f'hist_annealing_vs_cg_D={D}_L={type(loss).__name__}.jpeg', 400, overwrite = overwrite)