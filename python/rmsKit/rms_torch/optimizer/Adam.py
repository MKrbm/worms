from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _stack_if_compiling,
                        _dispatch_sqrt, _default_to_fused_or_foreach, _capturable_doc,
                        _differentiable_doc, _foreach_doc, _fused_doc, _maximize_doc)
from torch.optim import Adam as _Adam
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

__all__ = ['Adam']


class Adam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: Optional[bool] = None):
        
        if weight_decay != 0:
            raise ValueError("weight_decay != 0 not supported")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=0, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable, fused=fused)
        super().__init__(params, **defaults)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            custom_adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                fused=group['fused'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def custom_adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         foreach: Optional[bool] = None,
         capturable: bool = False,
         differentiable: bool = False,
         fused: Optional[bool] = None,
         grad_scale: Optional[Tensor] = None,
         found_inf: Optional[Tensor] = None,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         eps: float,
         maximize: bool):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    foreach = True

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if fused and not torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with fused optimizers')
    elif foreach and not torch.jit.is_scripting():
        # raise RuntimeError('torch.jit.script not supported with foreach optimizers')
        func = _multi_tensor_adam
    else:
        # func = _single_tensor_adam
        raise RuntimeError('torch.jit.script not supported for single tensor optimizers')

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         differentiable=differentiable,
         grad_scale=grad_scale,
         found_inf=found_inf)


def _multi_tensor_adam(params: List[Tensor],
                       grads: List[Tensor],
                       exp_avgs: List[Tensor],
                       exp_avg_sqs: List[Tensor],
                       max_exp_avg_sqs: List[Tensor],
                       state_steps: List[Tensor],
                       grad_scale: Optional[Tensor],
                       found_inf: Optional[Tensor],
                       *,
                       amsgrad: bool,
                       beta1: float,
                       beta2: float,
                       lr: float,
                       eps: float,
                       maximize: bool,
                       capturable: bool,
                       differentiable: bool):
    if len(params) == 0:
        return

    if capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for (device_params, device_grads, device_exp_avgs, device_exp_avg_sqs,
         device_max_exp_avg_sqs, device_state_steps) in grouped_tensors.values():

        if maximize:
            device_grads = torch._foreach_neg(tuple(device_grads))  # type: ignore[assignment]

        # Handle complex parameters
        if isinstance(device_params[0], int):
            raise RuntimeError("device_params should be a list of Tensors")
        if torch.is_complex(device_params[0]):
            raise 

        device_grads = [x for x in device_grads]
        device_exp_avgs = [x for x in device_exp_avgs]
        device_exp_avg_sqs = [x for x in device_exp_avg_sqs]
        params_ = [x for x in device_params]

        # update steps
        torch._foreach_add_(device_state_steps, 1)


        # Decay the first and second moment running average coefficient
        torch._foreach_mul_(device_exp_avgs, beta1)
        torch._foreach_add_(device_exp_avgs, device_grads, alpha=1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        if capturable:
            # TODO: use foreach_pow if/when foreach_pow is added
            bias_correction1 = [torch.pow(beta1, step) for step in device_state_steps]
            bias_correction2 = [torch.pow(beta2, step) for step in device_state_steps]
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            torch._foreach_neg_(bias_correction1)
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            step_size = torch._foreach_div(bias_correction1, lr)
            torch._foreach_reciprocal_(step_size)
            torch._foreach_neg_(step_size)

            bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)  # type: ignore[assignment]

                # Use the max. for normalizing running avg. of gradient
                max_exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                torch._foreach_div_(max_exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
                eps_over_step_size = torch._foreach_div(step_size, eps)
                torch._foreach_reciprocal_(eps_over_step_size)
                denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
                torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
                eps_over_step_size = torch._foreach_div(step_size, eps)
                torch._foreach_reciprocal_(eps_over_step_size)
                denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)

            foreacth_addcdiv_(params_, device_exp_avgs, denom)
        else:
            bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                max_exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
                torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
                denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
                torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
                denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

            foreacth_addcdiv_(params_, device_exp_avgs, denom, step_size)



def foreacth_addcdiv_(params, device_exp_avgs, denom, step_size=None):
    if step_size is not None:
        for param, exp_avg, d, s in zip(params, device_exp_avgs, denom, step_size):
            # param.addcdiv_(exp_avg, d, value=s)
            param.data[:] = torch.matrix_exp(s*exp_avg / d) @ param.data[:]
    else:
        for param, exp_avg, d in zip(params, device_exp_avgs, denom):
            param.data[:] = torch.matrix_exp(exp_avg / d) @ param.data[:]
            # param.addcdiv_(exp_avg, d)