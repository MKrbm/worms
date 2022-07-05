import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import SGD, Adam
from typing import List, Optional, Tuple
import abc
from scipy import optimize
from nsp.utils.func import type_check
from ..model import UnitaryRiemanGenerator

class BaseRiemanOptimizer(Optimizer, abc.ABC):
    """
    base class for rieman optimizatino for unitary matrix. input argument takes model instead of params, since riemanian grad use the model.matrix()
    """
    model : UnitaryRiemanGenerator

    def __init__(self,  model : UnitaryRiemanGenerator, lr,momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        self.model = model

        super(BaseRiemanOptimizer, self).__init__(model.parameters(), defaults)

        if (len(self.param_groups) != 1):
            raise ValueError("Length of self.param_groups must be 1")
        


    def _riemannian_grad(self, params, translated=True):
        """
        The geodesic emanating from W(U(D) embeded in euclid space) in the direction \tilde{S} = S W is given by
        G(t) = exp(-tS)W, S \in \mathfrak(u)(D), t \in \mathcal{R}

        Args:
            params : torch.Tensor with gradient
            translated : if True, return S insted of \tilde{S}
        """

        if not hasattr(params, "grad"):
            raise AttributeError("params should be a tensor with gradient")

        U = self.model.matrix()
        if not self._check_is_unitary(U):
            raise ValueError("U becomses non-unitary matrix")
        euc_grad = params.grad.view([self.model.D]*2)
        if translated: 
            #Gradient translated to the group identity 
            riemannianGradient = euc_grad @ U.T.conj() -U @ euc_grad.T.conj()
        else: 
            #Gradient in the tangent space of `orth`
            riemannianGradient = euc_grad - U @ euc_grad.T.conj() @ U 

        self.riemannianGradient = riemannianGradient
        self.U = U
        return riemannianGradient, U


    @torch.no_grad()
    def step(self, closure=None):
        # self._cuda_graph_capture_health_check()
        loss = None
        for group in self.param_groups:
            params_with_grad = []
            rd_p_n_U_list = [] #riemannian gradient and its euclidean coordinate in D by D matrix
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    rd_p_n_U_list.append(self._riemannian_grad(p, True)) 
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self.method(
                params_with_grad,
                rd_p_n_U_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,)
                
            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
    
    @staticmethod
    @abc.abstractmethod
    def method(
            params: List[torch.Tensor],
            rd_p_n_U_list: List[torch.Tensor],
            momentum_buffer_list: List[Optional[torch.Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool,
            maximize: bool):
        """
        calculate one step for params from rieman gradient and original params.
        """

    def _check_is_unitary(self, U):
        return (torch.round(U @ self.model._inv(U), decimals=10) == torch.eye(U.shape[0])).all()


class RiemanSGD(BaseRiemanOptimizer):
    

    @staticmethod
    def method(
            params: List[torch.Tensor],
            rd_p_n_U_list: List[Tuple[torch.Tensor, torch.Tensor]],
            momentum_buffer_list: List[Optional[torch.Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool,
            maximize: bool):
        """
        corresponds to sgd.
        """
        for i, param in enumerate(params):
            rd_p, U = rd_p_n_U_list[i] 

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(rd_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(rd_p, alpha=1 - dampening)
                
                if nesterov:
                    rd_p = rd_p.add(buf, alpha=momentum)
                else:
                    rd_p = buf

            alpha = lr if maximize else -lr
            param.data = (torch.matrix_exp(rd_p * alpha) @ U).view(-1)


class RiemanCG(BaseRiemanOptimizer):
    
    def __init__(self,  model : UnitaryRiemanGenerator, loss, lr,momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False):
        super().__init__(model,lr,momentum, dampening, weight_decay, nesterov, maximize=maximize)
        self.loss = loss

    def golden(self, U, H):
        
        if not (type_check(U) == type_check(H) == torch.Tensor) \
            or (U.requires_grad) \
            or (H.requires_grad):
            raise TypeError("type of U and H are required to be torch tensor without grad")

        def objective(t):
            return self.loss(torch.matrix_exp(-t*H)@U).item()

        return optimize.golden(objective, brack=(0, 1))


    def method(
            self,
            params: List[torch.Tensor],
            rd_p_n_U_list: List[Tuple[torch.Tensor, torch.Tensor]],
            momentum_buffer_list: List[Optional[torch.Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool,
            maximize: bool):
        """
        corresponds to sgd.
        """
        for i, param in enumerate(params):
            rd_p, U = rd_p_n_U_list[i] 
            rieman_grad_norm = (torch.trace(rd_p.T.conj() @ rd_p).real).item()
            if momentum_buffer_list[i] is None:
                lr = self.golden(U.data, rd_p.data)
                param.data = (torch.matrix_exp(rd_p * -lr) @ U).view(-1)
                momentum_buffer_list[i] = [torch.clone(rd_p.data), torch.clone(rd_p.data), rieman_grad_norm]
            else:

                [old_rd_p, old_norm] = momentum_buffer_list[i]
                curv_ratio = np.trace((rd_p-old_rd_p).T.conj()@rd_p).real / old_norm 
                inv_step_dir = rd_p+curv_ratio*old_rd_p
                lr = self.golden(U.data, inv_step_dir.data)
                param.data = (torch.matrix_exp(inv_step_dir * -lr) @ U).view(-1)
                momentum_buffer_list[i] = [
                    torch.clone(inv_step_dir.data), 
                    torch.clone(rd_p.data),
                    rieman_grad_norm
                ]