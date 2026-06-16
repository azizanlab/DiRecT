from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import logging
import omegaconf
from omegaconf import OmegaConf
from einops import rearrange, reduce
from copy import deepcopy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.projection.projection_operator import BaseProjectionOperator
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply

logger = logging.getLogger(__name__)

class DiffusionUnetLowdimProjPolicy(DiffusionUnetLowdimPolicy):
    def __init__(self, 
                projector: BaseProjectionOperator, 
                project_params: Dict = None, 
                timeit: bool = False,
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs)
        
        self.projector = projector
        if project_params is None or isinstance(project_params, dict):
            self.project_params = project_params if project_params is not None else {}
        else:
            assert isinstance(project_params, omegaconf.dictconfig.DictConfig) 
            self.project_params = OmegaConf.to_container(project_params, resolve=True)

        for k, v in self.project_params.items():
            self.project_params[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device)

        self.timeit = timeit
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask, 
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
            
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        # restore observation
        if self.obs_as_local_cond:
            obs = local_cond
        elif self.obs_as_global_cond:
            # condition throught global feature
            obs = global_cond.reshape(global_cond.shape[0], self.n_obs_steps, -1)
        else:
            obs = condition_data[condition_mask]
        assert obs.ndim == 3

        # set projection parameters 
        ### Attention! obs and action uses different normalizers. 
        ### so projection parameters we need to use the one from the action normalizer!  
        obs_unnormed = self.normalizer['obs'].unnormalize(obs.detach().clone())
        nobs_in_nact = self.normalizer['action'].normalize(
            obs_unnormed[:, :, -self.action_dim:]  # only the last action_dim elements are used for projection,
        )
        proj_params = {
            'pos_init': nobs_in_nact[:, -1, :], # initial position
            'dx_max': self.project_params['vel_max'] * self.project_params['delta_t'],
        }

        # pdb.set_trace()
        if self.timeit:
            tbench = {}

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, 
                global_cond=global_cond
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
            

            # 4. project trajectory onto sets 
            h_start = self.n_obs_steps - int(self.oa_step_convention) 
            traj_to_proj = trajectory[..., h_start:, :]
            if self.timeit:
                projected_traj, tbench_this_step = self.projector.project(
                    traj_to_proj, 
                    params = proj_params,
                    timeit=True,
                )
                if len(tbench) == 0:
                    tbench.update(tbench_this_step)
                else:
                    for k, v in tbench_this_step.items():
                        if k in tbench:
                            tbench[k] = np.concatenate([tbench[k], v], axis=0)
                        else:
                            tbench[k] = v
            else:
                projected_traj = self.projector.project(
                    traj_to_proj, 
                    params = proj_params, 
                )
            ## We put back the projected part without modifying the first n_obs_steps(-1) steps 
            ## which will be discarded later anyway 
            trajectory[..., h_start:, :] = projected_traj
            ## Optionally, we can do inpainting on the first n_obs_steps(-1) steps 
            ## this might provide a "smoother" trajectory but not the one that is partially noisy and partially projected
            if h_start > 0:
                trajectory[..., :h_start, :] = nobs_in_nact[:, -h_start:,:]

            # pdb.set_trace()
            pass 
        #end for t


        if self.timeit:
            tbench = dict_apply(
                tbench, 
                lambda _x: _x.mean()
            )
            t_sum = tbench['total'].mean()
            for k, v in tbench.items():
                logger.info(f"Projection time {k}: {v:.4f} s ({v / t_sum:.2%} of total)")

        self.sanity_check(projected_traj, proj_params['pos_init'], proj_params['dx_max'])

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask] 

        return trajectory
    
    def sanity_check(self, traj, x0, d_max, eps=1e-6):
        with torch.no_grad():
            dx0 = traj[:, 0, :] - x0
            dx0_norm = torch.linalg.norm(dx0, dim=-1)
            violation:torch.Tensor = dx0_norm - d_max
            max_violation = violation.max().item()
        if max_violation > eps:
            logger.warning(
                f"Trajectory contains steps exceeding max distance {d_max}; "
                f" Max violation: {max_violation:.3e} (> {eps:.3e})"
            )
            return False
        else:
            return True
        

    ## For debug only! 
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result
