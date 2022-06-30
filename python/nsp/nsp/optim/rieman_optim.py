import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import SGD
from typing import List, Optional, Tuple
import abc
from ..model import UnitaryRiemanGenerator

class BaseRiemanOptimizer(Optimizer, abc.ABC):
    """
    base class for rieman optimizatino for unitary matrix. input argument takes model instead of params, since riemanian grad use the model.matrix()
    """
    model : UnitaryRiemanGenerator

    def __init__(self,  model : UnitaryRiemanGenerator, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        self.model = model

        super(BaseRiemanOptimizer, self).__init__(model.parameters(), defaults)

        if (len(self.param_groups) != 1):
            raise ValueError("Length of self.param_groups must be 1")

    def _riemannian_grad(self,params, translated=True):
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
        euc_grad = params.grad.view([self.model.D]*2)
        if translated: 
            #Gradient translated to the group identity 
            riemannianGradient = euc_grad @ U.T.conj() -U @ euc_grad.T.conj()
        else: 
            #Gradient in the tangent space of `orth`
            riemannianGradient = euc_grad - U @ euc_grad.T.conj() @ U 

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

        for i, param in enumerate(params):
            rd_p, U = rd_p_n_U_list[i] 
            invStepDir = rd_p
            param.data = (torch.matrix_exp(invStepDir * -lr) @ U).view(-1)
