from copy import deepcopy
from enum import unique
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


class BaseRiemanNonTransUnitaryOptimizer(abc.ABC):
    """
    base class for rieman optimizatino for unitary matrix. input argument takes model instead of params, since riemanian grad use the model.matrix()

    args :
        models_act : 
    """
    models_act : List[Tuple[UnitaryRiemanGenerator]]
    models : List[UnitaryRiemanGenerator]
    def __init__(self,
                models_act : List[Tuple[UnitaryRiemanGenerator]], loss_list : List[BaseMatirxLoss], 
                lr = 0.01, 
                momentum=0, 
                weight_decay=0, 
                grad_tol = 1e-8, 
                *, 
                pout = False):

        # defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
        #                 weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        if (isinstance(models_act, UnitaryRiemanGenerator)):
            models_act = [models_act]
        models_act = list(models_act)
        self.models = []
        unique_ids = []
        for i, models in enumerate(models_act):
            if (isinstance(models, UnitaryRiemanGenerator)):
                print("Warning! model should be a list tuple of model")
                models_act[i] = (models,)
            models_act[i] = list(models_act[i])
            for model in models_act[i]:
                if id(model) not in unique_ids:
                    self.models.append(model)
                    unique_ids.append(id(model))
         
        self.models_act = models_act
        for models in self.models_act:
            for i in range(len(models)):
                models[i] = unique_ids.index(id(models[i]))
                
        self.ids = unique_ids
        self.pout = pout
        self.momentum_buffer = [{} for _ in range(len(self.models))]

        if (isinstance(loss_list, BaseMatirxLoss)):
            self.loss_list = [loss_list]
        elif not (isinstance(loss_list[0], BaseMatirxLoss)):
            raise TypeError("model should be a list of model")
        else:
            self.loss_list = loss_list

        if (len(self.loss_list) != len(self.models_act)):
            raise ValueError("length of loss_list and models should be same")
        self.grad_tol = grad_tol
        # super(BaseRiemanUnitaryOptimizer, self).__init__(model.parameters(), defaults)

        if pout:
            print("number of models is : ",len(self.models))
        
    def loss_val(self, models = None):
        """
        calculate loss 
        """

        if models is None:
            models = self.models
            loss_ = torch.zeros(1, dtype=torch.float64)
            for acts, loss in zip(self.models_act, self.loss_list):
                loss_ += loss([models[a].matrix() for a in acts])
            return loss_

        else:
            loss_ = torch.zeros(1, dtype=torch.float64)
            for acts, loss in zip(self.models_act, self.loss_list):
                loss_ += loss([models[a] for a in acts])
            return loss_
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

        U = model.matrix().data
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
            self.momentum_buffer[i]['momentum_buffer'] = momentum_buffer

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
            self,
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


class RiemanNonTransUnitarySGD(BaseRiemanNonTransUnitaryOptimizer):

    def __init__(self,
                models_act : List[Tuple[UnitaryRiemanGenerator]], loss_list : List[BaseMatirxLoss], 
                lr = 0.01,
                momentum=0, 
                weight_decay=0, 
                grad_tol = 1e-8, 
                *, 
                pout = False):
        super().__init__(models_act, loss_list, lr = lr, momentum = momentum)

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
            momentum_buffer_list[i] = torch.clone(rd_p.data)
            if momentum_buffer_list[i] is not None:
                old_rd_p = momentum_buffer_list[i]
                inv_step_dir = rd_p+momentum*old_rd_p
                param.data = (torch.matrix_exp(inv_step_dir * -lr) @ U).view(-1)
        return False

class RiemanNonTransUnitaryCG(BaseRiemanNonTransUnitaryOptimizer):

    def __init__(self,
                models_act : List[Tuple[UnitaryRiemanGenerator]], loss_list : List[BaseMatirxLoss], 
                momentum=0, 
                weight_decay=0, 
                grad_tol = 1e-8, 
                *, 
                pout = False):
        super().__init__(models_act, loss_list, lr = 0.01)

    def _golden(self, grad_U_list, delta=0.001):
        
        for H, U in grad_U_list:
            if not (type_check(U) == type_check(H) == torch.Tensor) \
                or (U.requires_grad) \
                or (H.requires_grad):
                raise TypeError("type of U and H are required to be torch tensor without grad")

        def objective(t):
            U_list = []
            for H, U in grad_U_list:
                U_list.append(torch.matrix_exp(-t*H)@U)
            return self.loss_val(U_list)


        t = torch.tensor([0.], requires_grad=True)
        f = objective(t)
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
        return 0


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

        norm = 0
        curv = 0
        gamma = 0

        for i, param in enumerate(params):
            rd_p, U = rd_p_n_U_list[i] 
            if momentum_buffer_list[i] is None:
                pass
            else:
                [old_inv_step_dir, old_rd_p, old_norm] = momentum_buffer_list[i]
                curv += np.trace((rd_p-old_rd_p).T.conj()@rd_p).real
                # if i == 0:
                #     print((old_rd_p[0,:10]))
                norm += old_norm
        if (.5*norm < self.grad_tol):
            pass
        else:
            gamma = curv/norm
        self.inv_list = []
        for i, param in enumerate(params):
            rd_p, U = rd_p_n_U_list[i] 

            rieman_grad_norm = (torch.trace(rd_p.T.conj() @ rd_p).real).item()
            if momentum_buffer_list[i] is None:
                self.inv_list.append((rd_p, U.detach()))
                momentum_buffer_list[i] = [torch.clone(rd_p.data), torch.clone(rd_p.data), rieman_grad_norm]
            else:
                [old_inv_step_dir, _, _] = momentum_buffer_list[i]
                # print("derivative = ", rd_p)
                inv_step_dir = rd_p+gamma*old_inv_step_dir
                inv_step_dir = (inv_step_dir - inv_step_dir.H)/2
                self.inv_list.append((inv_step_dir, U.detach()))

                # add old information to buffer
                momentum_buffer_list[i] = [
                    torch.clone(inv_step_dir.data), 
                    torch.clone(rd_p.data),
                    rieman_grad_norm
                ]

        lr = self._golden(self.inv_list)

        for i, param in enumerate(params):
            inv_step_dir, U = self.inv_list[i]
            param.data = (torch.matrix_exp(inv_step_dir * -lr) @ U).view(-1)
        return False