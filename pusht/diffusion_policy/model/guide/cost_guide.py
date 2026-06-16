import torch
import torch.nn as nn
import pdb
import omegaconf
from omegaconf import OmegaConf
from typing import Optional, Callable, Union, Tuple, List


class CostGuide(nn.Module):
    """
    Posterior sampling guide for the Diffusion Policy.
    
    The guide uses a cost function to evaluate the cost. 
    """
    def __init__(self, cost_func: Callable=None, cost_func_param: Optional[dict] = None):
        super().__init__()
        self.cost_fn = cost_func
        if cost_func_param is None or isinstance(cost_func_param, dict):
            self.cost_func_param = cost_func_param if cost_func_param is not None else {}
        else:
            assert isinstance(cost_func_param, omegaconf.dictconfig.DictConfig) 
            self.cost_func_param = OmegaConf.to_container(cost_func_param, resolve=True)

    def set_cost_function(self, cost_func: Callable):
        """Set the cost function to be used by the guide.
        
        Args:
            cost_func (Callable): The cost function to be used.
        """
        self.cost_fn = cost_func

    def get_cost_function(self) -> Optional[Callable]:
        """Get the current cost function used by the guide.
        
        Returns:
            Optional[Callable]: The current cost function, or None if not set.
        """
        return self.cost_fn

    def forward(self, input_vars:Union[Tuple, List], *args, **kwargs):
        # normalized = kwargs.get("normalized", False)
        # custom_normalizer = kwargs.get("custom_normalizer", None)
        c = self.cost_fn(
            *input_vars, 
            # normalized=normalized, 
            # custom_normalizer=custom_normalizer
            **self.cost_func_param,
        )
        return c 

    def gradients(self, input_vars:Union[Tuple, List], *args, **kwargs):
        """Compute gradients of the cost function with respect to input variables.
        Args:
            input_vars: Input variables to compute the cost function.
            *args: Variable length argument list passed to the cost function.
            **kwargs: Arbitrary keyword arguments including:
                normalized (bool, optional): Whether to use normalized inputs. Defaults to False.
                custom_normalizer (optional): Custom normalizer to apply. Defaults to None.
                with_respect_to (optional): Variables to compute gradients with respect to. 
                    Defaults to input_vars.
        Returns:
            tuple: A tuple containing:
                - cost (torch.Tensor): The computed cost values.
                - tuple: Gradients with respect to the specified variables.
        Note:
            The function computes the sum of the cost tensor before computing gradients,
            and allows unused variables in the gradient computation.
        """
        
        normalized = kwargs.get("normalized", False)
        custom_normalizer = kwargs.get("custom_normalizer", None)
        wrt = kwargs.get("with_respect_to", input_vars)
        cost:torch.Tensor = self.forward(
            input_vars, 
            normalized=normalized, 
            custom_normalizer=custom_normalizer
        )
        c_sum = cost.sum()
        # grads = []
        # for i, rv in enumerate(input_vars):
        #     grads.append(torch.autograd.grad(c_sum, rv)[0])
        grads = torch.autograd.grad(c_sum, wrt)

        return cost, grads

