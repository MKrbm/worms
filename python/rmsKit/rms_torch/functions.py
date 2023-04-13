
#* below for torch
import torch
#* Type hinting change from Array to torch.Tensor

def check_is_unitary_torch(X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        I_prime = X @ X.T.conj()
        identity_matrix = torch.eye(X.shape[0], device=X.device)
        return (torch.abs(I_prime - identity_matrix) < 1e-8).all()


def riemannian_grad_torch(u: torch.Tensor, euc_grad: torch.Tensor) -> torch.Tensor:
    if not check_is_unitary_torch(u):
        raise ValueError("u must be unitary matrix")
    if u.dtype != euc_grad.dtype:
        raise ValueError("dtype of u and euc_grad must be same")

    if u.shape != euc_grad.shape:
        raise ValueError("shape of u and euc_grad must be same")
    rg = euc_grad @ u.T.conj() - u @ euc_grad.T.conj()
    return (rg - rg.T.conj()) / 2

