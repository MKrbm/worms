from header import *

x = np.random.randn(4,4)
x = (x+x.T.conj())/2
x=torch.tensor(x)
loss = nsp.loss.L1(x, [4])
# loss_l2 = nsp.loss.L2(x, [4, 4])
# loss_mes = nsp.loss.MES(x, [4, 4])

res = []
for t in range(10000):
    x = np.random.randn(4,4) + 1j*np.random.randn(4,4)
    x = (x+x.T.conj())/2
    x=torch.tensor(x)
    loss = nsp.loss.L1(x, [4])
    model = nsp.model.UnitaryRiemanGenerator(4, dtype=torch.complex128)
    sgd = RiemanSGD(model, 0.001)
    loss_old = loss(model.matrix()).item()
    loss(model.matrix()).backward()
    sgd.step()
    loss_new = loss(model.matrix()).item()
    res.append((loss_new - loss_old))

fig, ax = plt.subplots()
ax.hist(res, bins=100)
ax.set_title('diff in the direction of rieman gradient, 1000 samples/lr = 0.001')
ax.set_xlabel('diff')
ax.set_ylabel('freq')
ax.legend()
fig.show()
save_fig(plt, "images", 'l1_rieman_grad_complex_lr=0.001.jpg', dpi=400, overwrite=False)
