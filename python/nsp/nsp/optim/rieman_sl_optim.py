import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import SGD, Adam
from typing import List, Optional, Tuple
import abc
from scipy import optimize
from nsp.model.similarity_model import SlRiemanGenerator
from nsp.utils.func import type_check
from ..model import UnitaryRiemanGenerator
from ..loss.base_class import BaseMatirxLoss
from ..utils.func import *


class BaseRiemanSlGOptimizer(Optimizer, abc.ABC):
    """
    base class for rieman optimizatino for special linear group. input argument takes model instead of params, since riemanian grad use the model.matrix()
    """
    model : SlRiemanGenerator
    def __init__(self,
                model : SlRiemanGenerator, lr, 
                momentum=0, dampening=0,weight_decay=0, 
                nesterov=False, *, maximize=False, pout = False):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        self.model = model
        self.pout = pout

        super(BaseRiemanSlGOptimizer, self).__init__(model.parameters(), defaults)

        if (len(self.param_groups) != 1):
            raise ValueError("Length of self.param_groups must be 1")
        


    def _riemannian_grad(self, params, translated=True):
        """
        The geodesic emanating from W(SL(D) embeded in euclid space) in the direction \tilde{S} = S W is given by
        G(t) = exp(-tS)W, S \in \mathfrak(u)(D), t \in \mathcal{R}

        Args:
            params : torch.Tensor with gradient
            translated : if True, return S insted of \tilde{S}
        """

        if not hasattr(params, "grad"):
            raise AttributeError("params should be a tensor with gradient")

        SL = self.model.matrix().detach()
        euc_grad = self.model._get_matrix(params.grad.detach())
        if translated: 
            #Gradient translated to the group identity 
            riemannianGradient = SL @ SL.T.conj() @ euc_grad @ self.model._inv(SL)
        else: 
            #Gradient in the tangent space of `orth`
            riemannianGradient = SL @ SL.T.conj() @ euc_grad

        self.riemannianGradient = riemannianGradient/torch.linalg.norm(riemannianGradient)
        self.SL = SL
        return self.riemannianGradient, SL


    @torch.no_grad()
    def step(self, closure=None):
        """
        return True if fall into local minimum
        """
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

            if self.method(
                params_with_grad,
                rd_p_n_U_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,) is True:

                return True
                
            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        SL = self.model.matrix().detach()
        if not self._check_is_sl(SL):
            # print(SL @ SL.T.conj())
            if self.pout:
                print("The determinant of SL becomse not equal to 1 : {}".format(torch.det(SL).item()))
            SL /= abs(torch.det(SL.detach())) ** (1/SL.shape[0])
            self.model.set_params(SL.view(-1))
        
        return False
    
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

    def _check_is_sl(self, SL):
        SL = SL.detach()
        return abs(abs(torch.det(SL).item()) - 1) < 1E-7

class RiemanSlSGD(BaseRiemanSlGOptimizer):
    
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
            rd_p, SL = rd_p_n_U_list[i] 

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
            param.data = (torch.matrix_exp(rd_p * alpha) @ SL).view(-1)

class RiemanSlCG(BaseRiemanSlGOptimizer):
    
    loss : BaseMatirxLoss
    def __init__(self,  
        model : SlRiemanGenerator, loss : BaseMatirxLoss, 
        grad_tol = 1e-8, 
        *, maximize=False, pout = False):
        super().__init__(model, lr = 0, pout = pout)
        self.loss = loss
        self.grad_tol = grad_tol

    def _golden(self, SL, H):
        
        if not (type_check(SL) == type_check(H) == torch.Tensor) \
            or (SL.requires_grad) \
            or (H.requires_grad):
            raise TypeError("type of SL and H are required to be torch tensor without grad")

        def objective(t):
            return self.loss(torch.matrix_exp(-t*H)@SL).item()

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
            rd_p, SL = rd_p_n_U_list[i] 
            rieman_grad_norm = (torch.trace(rd_p.T.conj() @ rd_p).real).item()
            if (.5*rieman_grad_norm < self.grad_tol):
                return True
            # print(momentum_buffer_list[i] is None)
            if momentum_buffer_list[i] is None:
                lr = self._golden(SL.data, rd_p.data)
                param.data = (torch.matrix_exp(rd_p * -lr) @ SL).view(-1)
                momentum_buffer_list[i] = [torch.clone(rd_p.data), torch.clone(rd_p.data), rieman_grad_norm]
            else:
                [old_inv_step_dir, old_rd_p, old_norm] = momentum_buffer_list[i]
                curv_ratio = np.trace((rd_p-old_rd_p).T.conj()@rd_p).real / old_norm 
                # print("derivative = ", rd_p)
                inv_step_dir = rd_p+curv_ratio*old_inv_step_dir
                inv_step_dir = (inv_step_dir - inv_step_dir.H)/2
                lr = self._golden(SL.data, inv_step_dir.data)
                if (abs(lr) < 1e-10):
                    lr = 0
                    if abs(curv_ratio) < 1e-10:
                        return True
                param.data = (torch.matrix_exp(inv_step_dir * -lr) @ SL).view(-1)

                # add old information to buffer
                momentum_buffer_list[i] = [
                    torch.clone(inv_step_dir.data), 
                    torch.clone(rd_p.data),
                    rieman_grad_norm
                ]

        return False