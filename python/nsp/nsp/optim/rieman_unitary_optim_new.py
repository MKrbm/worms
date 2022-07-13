import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import SGD, Adam
from typing import List, Optional, Tuple
import abc
from scipy import optimize
from nsp.utils.func import type_check
from ..model import UnitaryRiemanGenerator
from ..loss.base_class import BaseMatirxLoss
from ..utils.func import *


class BaseRiemanUnitaryOptimizer_2(abc.ABC):
    """
    base class for rieman optimizatino for unitary matrix. input argument takes model instead of params, since riemanian grad use the model.matrix()
    """
    models : List[UnitaryRiemanGenerator]
    def __init__(self,
                models : List[UnitaryRiemanGenerator], 
                lr, 
                momentum=0, 
                weight_decay=0, 
                *, 
                pout = False):

        # defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
        #                 weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        self.lr = lr,
        self.momentum = momentum
        self.weight_decay = weight_decay
        if (isinstance(models, UnitaryRiemanGenerator)):
            self.models = [models]
        elif not (isinstance(models[0], UnitaryRiemanGenerator)):
            raise TypeError("model should be a list of model")
        else:
            self.models = models
        self.pout = pout
        self.momentum_buffer = [{}]*len(self.models)
        # super(BaseRiemanUnitaryOptimizer, self).__init__(model.parameters(), defaults)

        print("number of models is : ",len(self.models))
        


    def _riemannian_grad(self, params, model:UnitaryRiemanGenerator, translated=True):
        """
        The geodesic emanating from W(U(D) embeded in euclid space) in the direction \tilde{S} = S W is given by
        G(t) = exp(-tS)W, S \in \mathfrak(u)(D), t \in \mathcal{R}

        Args:
            params : torch.Tensor with gradient
            translated : if True, return S insted of \tilde{S}
        """

        if not hasattr(params, "grad"):
            raise AttributeError("params should be a tensor with gradient")

        U = model.matrix()
        if not self._check_is_unitary(U.detach(), model._inv):
            V, _, W = torch.linalg.svd(U.detach())
            model.set_params((V@W).view(-1))
        # euc_grad = params.grad.view([self.model.D]*2)
        euc_grad = model._get_matrix(params.grad.detach())
        if translated: 
            #Gradient translated to the group identity 
            riemannianGradient = euc_grad @ U.T.conj() -U @ euc_grad.T.conj()
        else: 
            #Gradient in the tangent space of `orth`
            riemannianGradient = euc_grad - U @ euc_grad.T.conj() @ U 

        self.riemannianGradient = riemannianGradient
        self.U = U
        return riemannianGradient, U


    # @torch.no_grad()
    def step(self, closure=None):
        """
        return True if fall into local minimum
        """
        # self._cuda_graph_capture_health_check()
        loss = None
        params_with_grad = []
        rd_p_n_U_list = [] #riemannian gradient and its euclidean coordinate in D by D matrix
        momentum_buffer_list = []
        for model, state in zip(self.models, self.momentum_buffer):
            with torch.no_grad():
                p = model._params
                if p.grad is not None:
                    params_with_grad.append(p)
                    rd_p_n_U_list.append(self._riemannian_grad(p, model, True)) 
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

        if self.method(
            params_with_grad,
            rd_p_n_U_list,
            momentum_buffer_list,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            lr=self.lr,) is True:

            return True
                
        # update momentum_buffers in state
        for i, momentum_buffer in enumerate(momentum_buffer_list):
            state = self.momentum_buffer[i]
            state['momentum_buffer'] = momentum_buffer

        for model in self.models:
            U = model.matrix().detach()
            if not self._check_is_unitary(U):
                # print(U @ U.T.conj())
                if self.pout:
                    print("U becomes non-unitary matrix")
                V, _, W = torch.linalg.svd(U)
                model.set_params((V@W).view(-1))
                # raise ValueError("U becomses non-unitary matrix")
        
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

    def _check_is_unitary(self, U, inv = cc):
        return is_identity_torch(U @ inv(U), U.dtype == torch.complex128)

    def zero_grad(self, set_to_none=False):
        for model in self.models:
            p = model._params
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

class RiemanUnitarySGD2(BaseRiemanUnitaryOptimizer_2):
    
    def method(
            self,
            params: List[torch.Tensor],
            rd_p_n_U_list: List[Tuple[torch.Tensor, torch.Tensor]],
            momentum_buffer_list: List[Optional[torch.Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            lr: float,):
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
                    buf.mul_(momentum).add_(rd_p, alpha=1)
                
                rd_p = buf

            alpha =  -lr
            param.data = (torch.matrix_exp(rd_p * alpha) @ U).view(-1)

class RiemanUnitaryCG2(BaseRiemanUnitaryOptimizer_2):
    
    loss : BaseMatirxLoss
    def __init__(self,  
        models : List[UnitaryRiemanGenerator], loss_list : List[BaseMatirxLoss], 
        grad_tol = 1e-8, 
        *, pout = False):
        super().__init__(models,lr = 0, pout=pout)
        if (isinstance(loss_list, BaseMatirxLoss)):
            self.loss_list = [loss_list]
        elif not (isinstance(loss_list[0], BaseMatirxLoss)):
            raise TypeError("model should be a list of model")
        else:
            self.loss_list = loss_list

        if (len(self.loss_list) != len(self.models)):
            raise ValueError("length of loss_list and models should be same")
        self.grad_tol = grad_tol

    def _golden(self, U, H, delta=0.001, i = 0):
        loss = self.loss_list[i]
        
        if not (type_check(U) == type_check(H) == torch.Tensor) \
            or (U.requires_grad) \
            or (H.requires_grad):
            raise TypeError("type of U and H are required to be torch tensor without grad")

        def objective(t):
            return loss(torch.matrix_exp(-t*H)@U).item()

        t = torch.tensor([0.], requires_grad=True)
        f = loss(torch.matrix_exp(-t*H)@U)
        g = torch.autograd.grad(f, t, create_graph=True)
        g2 = torch.autograd.grad(g, t, create_graph=True)
        with torch.no_grad():
            step = (g[0] / g2[0]) * delta
        step = abs(step.item())
        t = 10
        for _ in range(10):
            # print(objective(step))
            if (objective(step) < objective(0)):
                while True:
                    if objective(step) < objective(step*t):
                        # print(objective(0) ,objective(step), objective(step*t))
                        a = optimize.golden(objective, brack = (0, step, step*t))
                        return a
                    else:
                        step *= t
            else:
                step /= t
        print("No local minimum found")
        a = optimize.golden(objective)
        print(a, step, objective(0), objective(a))
        return a


    def method(
            self,
            params: List[torch.Tensor],
            rd_p_n_U_list: List[Tuple[torch.Tensor, torch.Tensor]],
            momentum_buffer_list: List[Optional[torch.Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            lr: float,):
        """
        corresponds to sgd.
        """
        for i, param in enumerate(params):
            rd_p, U = rd_p_n_U_list[i] 
            rieman_grad_norm = (torch.trace(rd_p.T.conj() @ rd_p).real).item()
            if (.5*rieman_grad_norm < self.grad_tol):
                return True
            # print(momentum_buffer_list[i] is None)
            if momentum_buffer_list[i] is None:
                lr = self._golden(U.data, rd_p.data, i = i)
                param.data = (torch.matrix_exp(rd_p * -lr) @ U).view(-1)
                momentum_buffer_list[i] = [torch.clone(rd_p.data), torch.clone(rd_p.data), rieman_grad_norm]
            else:
                [old_inv_step_dir, old_rd_p, old_norm] = momentum_buffer_list[i]
                curv_ratio = np.trace((rd_p-old_rd_p).T.conj()@rd_p).real / old_norm 
                # print("derivative = ", rd_p)
                inv_step_dir = rd_p+curv_ratio*old_inv_step_dir
                inv_step_dir = (inv_step_dir - inv_step_dir.H)/2
                lr = self._golden(U.data, inv_step_dir.data, i = i)
                if (abs(lr) < 1e-10):
                    lr = 0
                    if abs(curv_ratio) < 1e-10:
                        return True
                param.data = (torch.matrix_exp(inv_step_dir * -lr) @ U).view(-1)

                # np.set_printoptions(precision=10)
                if not (inv_step_dir + inv_step_dir.T.conj() == 0).all():
                    print("Warning! inv_step_dir is not a skew matrix")
                    print(inv_step_dir)

                # add old information to buffer
                momentum_buffer_list[i] = [
                    torch.clone(inv_step_dir.data), 
                    torch.clone(rd_p.data),
                    rieman_grad_norm
                ]

        return False