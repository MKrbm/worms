import torch
import numpy as np
import abc
import copy
from tqdm.auto import tqdm

from ..model.unitary_model import BaseMatrixGenerator, UnitaryRiemanGenerator
from ..loss.base_class import BaseMatirxLoss
from ..optim.rieman_optim import BaseRiemanOptimizer
from typing import Union

class BaseGs(abc.ABC):
    """
    base class for gradient solver
    """

    model : BaseMatrixGenerator
    loss : BaseMatirxLoss
    optim_method : Union[torch.optim.Optimizer, BaseRiemanOptimizer]
    def __init__(
            self,
            optim_method : Union[torch.optim.Optimizer, BaseRiemanOptimizer],
            model : BaseMatrixGenerator,
            loss : BaseMatirxLoss,
            seed = None,
            **kwargs
            ):
        
        self.model = model
        if seed:
            self.model.reset_params(seed)
        if issubclass(optim_method, BaseRiemanOptimizer):
            if not isinstance(self.model, UnitaryRiemanGenerator):
                raise TypeError("If you want to optimize with rieman generator, then you need to use rieman generator")
            self.optim = optim_method(self.model, **kwargs)
        else:
            if isinstance(self.model, UnitaryRiemanGenerator):
                raise TypeError("Model for Riemannian optimization is only be available with UnitaryRiemanGenerator and its variants")
            self.optim = optim_method(self.model.parameters(), **kwargs)
        self.loss = loss
        self.target = loss.target
        if not (self.model._type == torch.Tensor):
            raise TypeError("only torch.Tensor is available")
        if not (self.loss._type == torch.Tensor):
            print("loss._type is converted from np.ndarray to torch.Tensor")
            self.loss._type_convert(torch.Tensor)
        self.best_model = copy.deepcopy(self.model)

    @abc.abstractmethod
    def run(self, n_iter, message=True):
        """
        main function
        """
from collections import OrderedDict
class UnitarySymmTs(BaseGs):
    """
    transform X by unitary matrix which is represented as U \otimes U \otimes \cdots U[len(act)-1]
    assume all local unitary matrices are same.
    """
    def run(self, n_iter, disable_message=False):
        
        model = self.model
        loss = self.loss
        optim = self.optim

        interval = n_iter / 100
        min_loss = loss([model.matrix()]*loss._n_unitaries).item()
        ini_loss = min_loss
        if not disable_message:
            print("target loss      : {:.10f}".format(self.target))
            print("initial loss     : {:.10f}\n".format(ini_loss))
            print("="*50, "\n")
        # for t in tqdm(range(n_iter)):
        ret = {}
        with tqdm(range(n_iter), disable=disable_message) as pbar:
            for t, ch in enumerate(pbar):
                U = model.matrix()
                loss_ = loss(U)
                pbar.set_postfix(OrderedDict(iter=t, loss=loss_.item()))
                if (loss_.item() < min_loss or min_loss is None):
                    min_loss = loss_.item()
                    self.best_model.set_params(model._params, True)
                optim.zero_grad()
                loss_.backward()
                optim.step()
        ret["model"] = self.best_model
        ret["target_loss"] = self.target
        ret["best_loss"] = min_loss
        return ret

