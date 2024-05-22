import sys
import os
import torch
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt

PYTHON_DIR = Path(os.getcwd()).parent.resolve()
(
    sys.path.insert(0, PYTHON_DIR.as_posix())
    if PYTHON_DIR.as_posix() not in sys.path
    else None
)
from rmsKit import rms_torch, lattice

params = {
    "sps": 3,
    "rank": 2,
    "dimension": 1,
    "seed": 3,  # seed = 3 is insitefull
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
n_iter = 1000

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
p_vec = (
    random_vec
    - (torch.dot(random_vec, rg_flat) / torch.dot(rg_flat, rg_flat)) * rg_flat
)

p_rg = p_vec.view(rg.shape)
p_rg = p_rg - p_rg.H
P_RG = torch.kron(p_rg, I) + torch.kron(I, p_rg)

print(P_RG)
assert torch.isclose(torch.trace(P_RG @ RG), torch.tensor(0.0, dtype=dtype))


U = model.forward().clone().detach()

steps = np.arange(-10, 10, 0.1)


def landscape_loss(step):
    Uc = torch.matrix_exp(-step * RG) @ U
    return mel(Uc).clone().detach()


loss_vals = [landscape_loss(step) for step in steps]

# Plotting the loss landscape
plt.figure(figsize=(10, 6))
plt.plot(steps, loss_vals, label="Loss Landscape")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss Landscape along Tangent Vector rg")
plt.legend()
plt.grid(True)
plt.show()


def landscape_loss_perp(step):
    Uc = torch.matrix_exp(-step * P_RG) @ U
    return mel(Uc).clone().detach()


loss_vals = [landscape_loss_perp(step) for step in steps]

# Plotting the loss landscape
plt.figure(figsize=(10, 6))
plt.plot(steps, loss_vals, label="Loss Landscape")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss Landscape along Tangent Vector p_rg")
plt.legend()
plt.grid(True)
plt.show()


# 2D landscape plot


def loss_2d(x, y):
    rg_prime = x * RG + y * P_RG
    Uc = torch.matrix_exp(-rg_prime) @ U
    return mel(Uc).clone().detach()


# Generate a meshgrid for the 2D plane
x = torch.linspace(-3, 3, 100)
y = torch.linspace(-3, 3, 100)
X, Y = torch.meshgrid(x, y)

# Compute the loss values for the 2D plane
Z = torch.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = loss_2d(X[i, j], Y[i, j])

# Convert tensors to numpy arrays for plotting
X = X.numpy()
Y = Y.numpy()
Z = Z.numpy()

# Plotting the 3D loss landscape using plotly
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])

# Add a vertical line at (0, 0)
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
