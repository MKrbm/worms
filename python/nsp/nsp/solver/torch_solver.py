import torch
import numpy as np
import abc
import copy
from tqdm.auto import tqdm

from ..model.unitary_model import BaseMatrixGenerator
from ..loss.max_eig import BaseUlf


class BaseGs(abc.ABC):
    """
    base class for gradient solver
    """

    # model : BaseMatrixGenerator
    # loss : base_ulf
    def __init__(
            self,
            optim_method,
            model : BaseMatrixGenerator,
            loss : BaseUlf,
            seed = None,
            **kwargs
            ):
        
        self.model = model
        if seed:
            self.model.reset_params(seed)
        self.optim = optim_method(self.model.parameters(), **kwargs)
        self.loss = loss
        self.target = loss.target
        if not (self.loss._type == self.model._type == torch.Tensor):
            raise TypeError("only torch.Tensor is available")
        self.best_model = copy.deepcopy(self.model)

    @abc.abstractmethod
    def run(self):
        """
        main function
        """
from collections import OrderedDict
class UnitarySymmTs(BaseGs):
    """
    transform X by unitary matrix which is represented as U \otimes U \otimes \cdots U[len(act)-1]
    assume all local unitary matrices are same.
    """
    def run(self, n_iter):
        
        model = self.model
        loss = self.loss

        interval = n_iter / 100
        min_loss = loss([model.matrix()]*loss._n_unitaries).item()
        ini_loss = min_loss
        print("target loss      : {:.10f}".format(self.target))
        print("initial loss     : {:.10f}\n".format(ini_loss))
        print("="*50, "\n")
        # for t in tqdm(range(n_iter)):
        with tqdm(range(n_iter)) as pbar:
            for t, ch in enumerate(pbar):
                U = model.matrix()
                loss_ = loss(U)
                pbar.set_postfix(OrderedDict(iter=t, loss=loss_.item()))
                if (loss_.item() < min_loss or min_loss is None):
                    min_loss = loss_.item()
                    self.best_model.set_params(model._params)
                self.optim.zero_grad()
                loss_.backward()
                self.optim.step()
        


