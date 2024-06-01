import sys
import os
import torch
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

PYTHON_DIR = Path(os.getcwd()).parent.resolve()
sys.path.insert(0, PYTHON_DIR.as_posix()) if PYTHON_DIR.as_posix() not in sys.path else None
from rmsKit import rms_torch, lattice

params = {
    "sps": 3,
    "rank": 2,
    "dimension": 1,
    "seed": 3,
    "lt": 1,
}

dtype = torch.float64

h, sps = lattice.FF.local(params)
h_torch = torch.tensor(h)

model = rms_torch.UnitaryRiemann(h.shape[1], sps, dtype=dtype)

step = 0.01
optimizer = rms_torch.Adam(model.parameters(), lr=step, betas=(0.9, 0.999))

mel = rms_torch.MinimumEnergyLoss(h_torch, dtype=dtype)

step = 0.001
n_iter = 100

for i in range(n_iter):
    optimizer.zero_grad()
    U = model.forward()
    loss = mel(U)
    loss.backward()
    model.update_riemannian_gradient()
    optimizer.step()
    print(f"loss : {loss.item()}")

rg = (model.u[0].grad).clone().detach()
I = torch.eye(rg.shape[0])
RG = torch.kron(rg, I) + torch.kron(I, rg)
rg_flat = rg.flatten()
random_vec = torch.randn_like(rg_flat)
p_vec = random_vec - (torch.dot(random_vec, rg_flat) / torch.dot(rg_flat, rg_flat)) * rg_flat

p_rg = p_vec.view(rg.shape)
p_rg = p_rg - p_rg.H
P_RG = torch.kron(p_rg, I) + torch.kron(I, p_rg)

print(P_RG)
assert torch.isclose(torch.trace(P_RG @ RG), torch.tensor(0.0, dtype=dtype))

U = model.forward().clone().detach()

steps = np.arange(-10, 10, 0.1)

H_old = None
def positive_elems_rate(U, H):
    UHU = U @ H @ U.H

    global H_old
    if H_old is None:
        H_old = UHU.detach().clone()
    
    sign_change = (torch.sign(UHU) - torch.sign(H_old)).sum() / 2
    H_old = UHU.detach().clone()
    return torch.abs(sign_change)

def landscape_loss(step):
    Uc = torch.matrix_exp(-step * RG) @ U
    return mel(Uc).clone().detach(), positive_elems_rate(Uc, mel.h_tensor[0])

def landscape_loss_perp(step):
    Uc = torch.matrix_exp(-step * P_RG) @ U
    return mel(Uc).clone().detach(), positive_elems_rate(Uc, mel.h_tensor[0])

loss_vals, positive_elems_rates = zip(*[landscape_loss(step) for step in steps])

peaks, _ = find_peaks(-np.array(loss_vals))

fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss Landscape along Tangent Vector rg", "Positive Elements Rate along Tangent Vector rg"))

fig.add_trace(go.Scatter(x=steps, y=loss_vals, mode='lines', name='Loss Landscape'), row=1, col=1)
for peak in peaks:
    fig.add_vline(x=steps[peak], line=dict(color='red', dash='dash'), name='Peak', row=1, col=1)

fig.add_trace(go.Scatter(x=steps, y=positive_elems_rates, mode='lines', name='Positive Elements Rate'), row=2, col=1)
for peak in peaks:
    fig.add_vline(x=steps[peak], line=dict(color='red', dash='dash'), name='Peak', row=2, col=1)

fig.update_layout(
    title="Loss Landscape and Positive Elements Rate along Tangent Vector rg",
    xaxis_title="Steps",
    showlegend=True,
    # yaxis2=dict(matches='y'),
    xaxis2=dict(matches='x')
)

fig.update_yaxes(title_text="Loss", row=1, col=1)
fig.update_yaxes(title_text="Positive Elements Rate", row=2, col=1)

fig.show()

loss_vals, positive_elems_rates = zip(*[landscape_loss_perp(step) for step in steps])

peaks, _ = find_peaks(-np.array(loss_vals))

fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss Landscape along Tangent Vector p_rg", "Positive Elements Rate along Tangent Vector p_rg"))

fig.add_trace(go.Scatter(x=steps, y=loss_vals, mode='lines', name='Loss Landscape'), row=1, col=1)
for peak in peaks:
    fig.add_vline(x=steps[peak], line=dict(color='red', dash='dash'), name='Peak', row=1, col=1)

fig.add_trace(go.Scatter(x=steps, y=positive_elems_rates, mode='lines', name='Positive Elements Rate'), row=2, col=1)
for peak in peaks:
    fig.add_vline(x=steps[peak], line=dict(color='red', dash='dash'), name='Peak', row=2, col=1)

fig.update_layout(
    title="Loss Landscape and Positive Elements Rate along Tangent Vector p_rg",
    xaxis_title="Steps",
    showlegend=True,
    # yaxis2=dict(matches='y'),
    xaxis2=dict(matches='x')
)

fig.update_yaxes(title_text="Loss", row=1, col=1)
fig.update_yaxes(title_text="Positive Elements Rate", row=2, col=1)

fig.show()

def loss_2d(x, y):
    rg_prime = x * RG + y * P_RG
    Uc = torch.matrix_exp(-rg_prime) @ U
    return mel(Uc).clone().detach()

x = torch.linspace(-3, 3, 100)
y = torch.linspace(-3, 3, 100)
X, Y = torch.meshgrid(x, y)

Z = torch.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = loss_2d(X[i, j], Y[i, j])

X = X.numpy()
Y = Y.numpy()
Z = Z.numpy()

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])
fig.add_trace(
    go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[Z.min(), Z.max()],
        mode="lines",
        line=dict(color="red", width=5),
        name="Local Minima",
    )
)

fig.update_layout(
    title="3D Loss Landscape",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Loss"),
)

fig.show()
