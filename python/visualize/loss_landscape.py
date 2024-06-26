import sys
import os
import torch
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

PYTHON_DIR = Path(__file__).parents[1].resolve()
sys.path.insert(0, PYTHON_DIR.as_posix()) if PYTHON_DIR.as_posix() not in sys.path else None
from rmsKit import rms_torch, lattice

params = {
    "sps": 8,
    "rank": 2,
    "dimension": 1,
    "seed": 10,
    "lt": 1,
}

dtype = torch.float64

h, sps = lattice.FF.local(params)
h_torch = torch.tensor(h)

model = rms_torch.UnitaryRiemann(h.shape[1], sps, dtype=dtype)

step = 0.01
optimizer = rms_torch.Adam(model.parameters(), lr=step, betas=(0.9, 0.999))

mel = rms_torch.MinimumEnergyLoss(h_torch, dtype=dtype)

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

# Number of vectors to generate
K = 8  # Example value, set this to your desired number of vectors

# Initialize list to hold the orthogonal vectors, starting with rg_flat
orthogonal_vectors = [rg_flat]

# Generate and orthogonalize K random vectors
for _ in range(K):
    random_vec = torch.randn_like(rg_flat)
    for ortho_vec in orthogonal_vectors:
        # Project random_vec onto ortho_vec and subtract to make orthogonal
        random_vec -= (torch.dot(random_vec, ortho_vec) / torch.dot(ortho_vec, ortho_vec)) * ortho_vec
    # Normalize the vector (optional, for numerical stability)
    random_vec /= torch.norm(random_vec)
    orthogonal_vectors.append(random_vec)

# Convert orthogonal vectors back to the original shape and store them
P_RGs = []
for vec in orthogonal_vectors[1:]:  # Skip the first one as it's rg_flat
    p_rg = vec.view(rg.shape)
    p_rg = p_rg - p_rg.H  # Ensure skew-symmetry
    P_RG = torch.kron(p_rg, I) + torch.kron(I, p_rg)
    P_RGs.append(P_RG)
    assert torch.isclose(torch.trace(P_RG @ RG), torch.tensor(0.0, dtype=dtype))


U = model.forward().clone().detach()

steps = np.arange(-30, 30, 0.1)

H_old = None

def positive_elems_rate(U, H):
    UHU = U @ H @ U.H
    global H_old
    if H_old is None:
        H_old = UHU.detach().clone()
    sign_change = (torch.sign(UHU) - torch.sign(H_old)).abs().sum() / 2
    H_old = UHU.detach().clone()
    return torch.abs(sign_change)

def negative_elems_count(U, H):
    UHU = U @ H @ U.H
    return (UHU < 0).sum().item()

def landscape_loss(step, RG, U):
    Uc = torch.matrix_exp(-step * RG) @ U
    loss = mel(Uc).clone().detach()
    return loss, positive_elems_rate(Uc, mel.h_tensor[0]), negative_elems_count(Uc, mel.h_tensor[0])

def visualize_landscape(RG, U, steps, title : str):
    loss_vals, positive_elems_rates, negative_elems_counts = zip(*[landscape_loss(step, RG, U) for step in steps])
    
    peaks, _ = find_peaks(-np.array(loss_vals))

    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        "Loss Landscape", "Positive Elements Changes", "Negative Elements Count"))

    fig.add_trace(go.Scatter(x=steps, y=loss_vals, mode='lines', name='Loss Landscape'), row=1, col=1)
    for peak in peaks:
        fig.add_vline(x=steps[peak], line=dict(color='red', dash='dash'), name='Peak', row=1, col=1)

    fig.add_trace(go.Scatter(x=steps, y=positive_elems_rates, mode='lines', name='Positive Elements Changes'), row=2, col=1)
    for peak in peaks:
        fig.add_vline(x=steps[peak], line=dict(color='red', dash='dash'), name='Peak', row=2, col=1)

    fig.add_trace(go.Scatter(x=steps, y=negative_elems_counts, mode='lines', name='Negative Elements Count'), row=3, col=1)
    for peak in peaks:
        fig.add_vline(x=steps[peak], line=dict(color='red', dash='dash'), name='Peak', row=3, col=1)

    fig.update_layout(
        title=title,
        xaxis_title="Steps",
        showlegend=True,
        xaxis3=dict(matches='x')
    )

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Positive Elements Changes", row=2, col=1)
    fig.update_yaxes(title_text="Negative Elements Count", row=3, col=1)

    fig.show()

# Visualize landscape for RG
visualize_landscape(RG, U, steps, "Loss Landscape along steepest descent direction")

# Visualize landscape for P_RG
for k, P_RG in enumerate(P_RGs):
    visualize_landscape(P_RG, U, steps, f"Loss Landscape along orthogonal direction k = {k}")



# def loss_2d(x, y):
#     rg_prime = x * RG + y * P_RGs[0]
#     Uc = torch.matrix_exp(-rg_prime) @ U
#     return mel(Uc).clone().detach()

# x = torch.linspace(-3, 3, 100)
# y = torch.linspace(-3, 3, 100)
# X, Y = torch.meshgrid(x, y)

# Z = torch.zeros_like(X)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         Z[i, j] = loss_2d(X[i, j], Y[i, j])

# X = X.numpy()
# Y = Y.numpy()
# Z = Z.numpy()

# fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])
# fig.add_trace(
#     go.Scatter3d(
#         x=[0, 0],
#         y=[0, 0],
#         z=[Z.min(), Z.max()],
#         mode="lines",
#         line=dict(color="red", width=5),
#         name="Local Minima",
#     )
# )

# fig.update_layout(
#     title="3D Loss Landscape",
#     scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Loss"),
# )

# fig.show()
