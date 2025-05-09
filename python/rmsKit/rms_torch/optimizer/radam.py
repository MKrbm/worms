import torch
import torch.optim

from geoopt.optim.mixin import OptimMixin
from geoopt.tensor import ManifoldParameter, ManifoldTensor

__all__ = ["RiemannianAdam"]

def expmap_with_given_M(
    W: torch.Tensor,  # (n, p) Stiefel point, with W^T W = I_p
    M: torch.Tensor   # (n, n) skew-symmetric matrix, i.e., M^T = -M
) -> torch.Tensor:
    r"""
    Update W on the Stiefel manifold by:
      1) Extending W to a full n x n orthonormal matrix W_tilde whose first p columns equal W.
      2) Computing expM = exp(M).
      3) Updating W_tilde_new = expM @ W_tilde.
      4) Setting W_new = W_tilde_new[:, :p].
    
    Since M is skew-symmetric, exp(M) is orthogonal and the result stays on the Stiefel manifold.
    """
    n, p = W.shape
    assert M.shape == (n, n), "M must be (n,n)"
    if p < n:
        # Create a random matrix for the complementary space of shape (n, n-p)
        A = torch.randn(n, n - p, dtype=W.dtype, device=W.device)
        # Project out the components in the span of W so that A is in the complement of W:
        A = A - W @ (W.T @ A)
        # Perform QR on A to get an orthonormal basis for the complement:
        Q, _ = torch.linalg.qr(A)
        # Concatenate W and Q to form W_tilde:
        W_tilde = torch.cat([W, Q], dim=1)
    else:
        # If p == n, W is already square and orthonormal.
        W_tilde = W

    # Exponentiate M (note: if M is skew-symmetric, exp(M) is orthogonal)
    expM = torch.linalg.matrix_exp(M)
    # Update the full orthonormal matrix
    W_tilde_new = expM @ W_tilde
    # Extract the first p columns to remain on the Stiefel manifold
    W_new = W_tilde_new[:, :p]

    return W_new

class RiemannianAdam(OptimMixin, torch.optim.Adam):
    r"""
    Riemannian Adam with the same API as :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm (default: False)
    stabilize : int (optional)
        Stabilize parameters if they are off-manifold every ``stabilize``
        steps (default: ``None`` -- no stabilization)
    """
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                group["step"] += 1
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, "
                            "use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]
                    n, p = point.shape
                    # State initialization
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros(n, n, dtype=point.dtype, device=point.device)
                        state["exp_avg_sq"] = torch.zeros(n, n, dtype=point.dtype, device=point.device)
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros(n, n, dtype=point.dtype, device=point.device)
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Preprocess gradient to be skew-symmetric
                    grad = grad @ point.T.conj()
                    grad = (grad - grad.T.conj()) / 2

                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    norm = grad.abs() ** 2
                    exp_avg_sq.mul_(betas[1]).add_(norm, alpha=1 - betas[1])
                    bias_correction1 = 1 - betas[0] ** group["step"]
                    bias_correction2 = 1 - betas[1] ** group["step"]

                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    direction = exp_avg.div(bias_correction1) / (denom.add_(eps))

                    # Update point on the Stiefel manifold using our expmap:
                    new_point = expmap_with_given_M(point, -direction * learning_rate)
                    point.copy_(new_point)

                if group.get("stabilize", None) is not None and group["step"] % group["stabilize"] == 0:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            p.copy_(manifold.projx(p))
            exp_avg.copy_(manifold.proju(p, exp_avg))
