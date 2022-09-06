import torch
import numpy as np
import abc
import copy
from tqdm.auto import tqdm
from scipy.optimize import OptimizeResult

from nsp.optim.rieman_nontrans_unitary_optim import RiemanNonTransUnitaryCG, RiemanNonTransUnitarySGD

from ..model.unitary_model import BaseMatrixGenerator, UnitaryRiemanGenerator
from ..model.similarity_model import SlRiemanGenerator
from ..loss.base_class import BaseMatirxLoss
from ..optim.rieman_unitary_optim import BaseRiemanUnitaryOptimizer, RiemanUnitaryCG
from ..optim.rieman_sl_optim import BaseRiemanSlGOptimizer, RiemanSlCG
from ..utils.func import *
from typing import Union

from collections import OrderedDict
class UnitaryTransTs:
    """

    helper class for iterately optimize.

    transform X by translated symmetric unitary matrix which is represented as U \otimes U \otimes \cdots U
    assume all local unitary matrices are same.
    Note:
        - only one local unitary matrix and loss function are required. 
        - Because optim_method can take torch.sgd, model and loss are also required
        - You can optimize exactly same way with UnitaryNonTransTS class

    args:
        optim : optimize that 
        model : unitary matrix (model) that you wanna optimize
        loss : loss function you want to minimize.
        seed : if given, reset matrix with this seed
        prt : wether print out or not 
    """

    model : BaseMatrixGenerator
    loss : BaseMatirxLoss
    optim_method : Union[torch.optim.Optimizer, BaseRiemanUnitaryOptimizer, BaseRiemanSlGOptimizer]
    def __init__(
            self,
            optim_method : Union[torch.optim.Optimizer, BaseRiemanUnitaryOptimizer, BaseRiemanSlGOptimizer],
            model : BaseMatrixGenerator,
            loss : BaseMatirxLoss,
            seed = None,
            prt = True,
            **kwargs
            ):
        self.prt = prt
        self.model = model
        self.loss = loss
        self.target = loss.target
        if seed:
            self.model.reset_params(seed)
        if issubclass(optim_method, BaseRiemanUnitaryOptimizer):
            if not isinstance(self.model, UnitaryRiemanGenerator):
                raise TypeError("If you want to optimize with rieman generator, then you need to use rieman generator")
            
            if optim_method in [RiemanUnitaryCG, RiemanSlCG]:
                self.optim = optim_method(self.model, self.loss, pout = prt, **kwargs)
            else:
                self.optim = optim_method(self.model, pout = prt, **kwargs)

        elif issubclass(optim_method, BaseRiemanSlGOptimizer):
            if not isinstance(self.model, SlRiemanGenerator):
                raise TypeError("If you want to optimize with rieman generator, then you need to use rieman generator")
            
            if optim_method in [RiemanUnitaryCG, RiemanSlCG]:
                self.optim = optim_method(self.model, self.loss, pout = prt, **kwargs)
            else:
                self.optim = optim_method(self.model, pout = prt, **kwargs)

        else:
            if isinstance(self.model, UnitaryRiemanGenerator):
                raise TypeError("Model for Riemannian optimization is only be available with UnitaryRiemanGenerator and its variants")
            self.optim = optim_method(self.model.parameters(), **kwargs)

        if not (self.model._type == torch.Tensor):
            raise TypeError("only torch.Tensor is available")
        if not (self.loss._type == torch.Tensor):
            print("loss._type is converted from np.ndarray to torch.Tensor")
            self.loss._type_convert(torch.Tensor)
        self.best_model = copy.deepcopy(self.model)

    def run(self, n_iter, disable_message=False):
        
        model = self.model
        loss = self.loss
        optim = self.optim

        # interval = n_iter / 100
        min_loss = loss([model.matrix()]*loss._n_unitaries).item()
        ini_loss = min_loss
        
        if not disable_message: 
            print("target loss      : {:.10f}".format(self.target))
            print("initial loss     : {:.10f}".format(ini_loss))
            print("loss upper bound : {:.10f}\n".format(loss(torch.eye(model.D)))) # n* loss if you take identity matrix as a unitary matrix
            print("="*50, "\n")
        # for t in tqdm(range(n_iter)):
        ret = {}
        loss_old = 1E9
        loss_same_cnt = 0 #n* if optimize return the same loss max_cnt times in a row, then break the loop
        max_cnt = 15
        with tqdm(range(n_iter), disable=disable_message) as pbar:
            for t, ch in enumerate(pbar):
                U = model.matrix()
                loss_ = loss(U)
                if (abs(loss_.item() - loss_old)<1E-9):
                    loss_same_cnt += 1
                    if loss_same_cnt > max_cnt:
                        if self.prt:
                            print("stack in local minimum --> break loop")
                        break
                else:
                    loss_same_cnt = 0
                pbar.set_postfix_str("iter={}, loss={:.5f}".format(t,loss_.item()))
                if (loss_.item() < min_loss or min_loss is None):
                    min_loss = loss_.item()
                    self.best_model.set_params(model._params, True)
                optim.zero_grad()
                loss_.backward()
                if optim.step():
                    if self.prt:
                        print("stack in local minimum --> break loop")
                    break
                loss_old = loss_.item()
        ret["model"] = self.best_model
        ret["target"] = self.target
        ret["fun"] = min_loss
        return OptimizeResult(ret)




class UnitaryNonTransTs:
    """
    solver class for a list of local unitary matrix
    transform X by non translational symmetic unitary matrix which is represented as U[0] \otimes U[1] \otimes \cdots U[len(act)-1]
    Note: 
        - Optim is a instance of optimier method. Not class itself unlike UnitaryTransTS

    Args:
        optim : Instance of RiemanNonTransUnitaryCG or RiemanNonTransUnitarySGD
    """

    optim : Union[RiemanNonTransUnitaryCG, RiemanNonTransUnitarySGD]
    def __init__(
            self,
            optim : Union[RiemanNonTransUnitaryCG, RiemanNonTransUnitarySGD],
            seed = None,
            prt = True,
            **kwargs
            ):
        self.prt = prt
        if seed:
            set_seed(seed)
            for model in optim.models:
                model.reset_params()
        if not isinstance(optim, RiemanNonTransUnitaryCG) and  not isinstance(optim, RiemanNonTransUnitarySGD):
            raise TypeError("It only receive {} or {} as a type of optim".format(RiemanNonTransUnitaryCG, RiemanNonTransUnitarySGD))
        self.best_models = copy.deepcopy(optim.models)
        self.optim = optim

        self.target=0
        for loss in self.optim.loss_list:
            self.target += loss.target

    def run(self, n_iter, disable_message=False):
        
        optim = self.optim

        min_loss = self.optim.loss_val().item()
        ini_loss = min_loss
        eye = [torch.eye(model.D, dtype=torch.float64) for model in self.optim.models]
        lub = self.optim.loss_val(eye).item()
        if not disable_message:
            print("target loss      : {:.10f}".format(self.target))
            print("initial loss     : {:.10f}".format(ini_loss))
            print("loss upper bound : {:.10f}\n".format(lub))
            print("="*50, "\n")
        ret = {}
        loss_old = 1E9
        loss_same_cnt = 0
        max_cnt = 15
        with tqdm(range(n_iter), disable=disable_message) as pbar:
            for t, ch in enumerate(pbar):
                loss_ = self.optim.loss_val()
                if (abs(loss_.item() - loss_old)<1E-9):
                    loss_same_cnt += 1
                    if loss_same_cnt > max_cnt:
                        if self.prt:
                            print("stack in local minimum --> break loop")
                        break
                else:
                    loss_same_cnt = 0
                pbar.set_postfix_str("iter={}, loss={:.5f}".format(t,loss_.item()))
                if (loss_.item() < min_loss or min_loss is None):
                    min_loss = loss_.item()
                    self.best_models = copy.deepcopy(self.optim.models)
                optim.zero_grad()
                loss_.backward()
                if optim.step():
                    if self.prt:
                        print("stack in local minimum --> break loop")
                    break
                loss_old = loss_.item()
        ret["model"] = self.best_models
        ret["target"] = self.target
        ret["fun"] = min_loss
        return OptimizeResult(ret)

