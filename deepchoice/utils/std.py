import torch
import torch.nn as nn
from copy import deepcopy


def parameter_std(model: nn.Module, loss_fn: callable):
    # the model should have been trained and converged.
    state_dict = deepcopy(model.state_dict())

    shape, start, end = dict(), dict(), dict()
    
    param_list = list()
    s = 0
    # wrap state dict into a single one dimensional tensor.
    for k, v in state_dict.items():
        num_params = state_dict[k].numel()
        shape[k] = v.shape
        param_list.append(v.clone().view(-1,))
        start[k], end[k] = (s, s + num_params)
        s += num_params
    all_params = torch.cat(param_list)

    def func(input_tensor):
        # unwrap parameters.
        # recovered_state_dict = dict()
        for k in state_dict.keys():
            src = input_tensor[start[k]: end[k]]
            src = src.view(*shape[k])
            # recovered_state_dict[k] = src

            # TODO(Tianyu): this is horrible implementation.
            exec(f'del model.{k}')
            exec(f'model.{k} = src')
            
        # breakpoint()
        return loss_fn(model)

    H = torch.autograd.functional.hessian(func, all_params)

    std_all = torch.sqrt(torch.diag(torch.inverse(H)))
    std_dict = dict()
    for k in state_dict.keys():
        std_dict[k] = std_all[start[k]: end[k]].view(*shape[k]).detach().cpu()

    return std_dict
