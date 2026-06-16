import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import logging
import omegaconf
from omegaconf import OmegaConf
from einops import rearrange, reduce
from copy import deepcopy
from typing import Union, Dict, List, Optional
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.diffusion_unet_lowdim_cost_policy import DiffusionUnetLowdimCostPolicy
from diffusion_policy.model.projection.projection_operator import BaseProjectionOperator
from diffusion_policy.model.guide.cost_guide import CostGuide
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

logger = logging.getLogger(__name__)

class DiffusionUnetLowdimProjCoupPolicy(DiffusionUnetLowdimCostPolicy):
    def __init__(self,
                guide: CostGuide,
                projector: BaseProjectionOperator,
                project_params: Union[Dict, omegaconf.dictconfig.DictConfig],
                *args,
                n_guide_steps=1,
                grad_scale=1.,
                t_stopgrad = 2,
                threshold=1e-6,
                point_estimate=False,
                timeit=False,
                group_size=2,
                **kwargs):
        super().__init__(
            *args,
            guide=guide,
            n_guide_steps=n_guide_steps,
            grad_scale=grad_scale,
            t_stopgrad=t_stopgrad,
            threshold=threshold,
            point_estimate=point_estimate,
            group_size=group_size,
            **kwargs
        )
        ## Projection
        self.projector = projector
        self.project_params:Dict = {}
        if isinstance(project_params, dict):
            self.project_params = project_params 
        else:
            assert isinstance(project_params, omegaconf.dictconfig.DictConfig) 
            self.project_params = OmegaConf.to_container(project_params, resolve=True)

        for k, v in self.project_params.items():
            self.project_params[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device)

        assert not timeit, "Timeit is not supported in this policy yet."
        self.timeit = False

    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            guide=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
    )->torch.Tensor:
        model = self.model
        scheduler = self.noise_scheduler
        if guide is None:
            guide = self.guide
        threshold = self.threshold
        return_grads = kwargs.pop('return_grads', False)
        intermediates_log = kwargs.pop('intermediates_log', None)
        assert condition_data.shape[0] % self.group_size == 0, \
            f"Batch size {condition_data.shape[0]} must be divisible by group_size {self.group_size}"

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        ## Prepare trajectory gradients
        if return_grads:
            guidance_gradients:List[torch.Tensor] = []

        ## -------------------- Projection Preparation -------------------
        # restore observation
        if self.obs_as_local_cond:
            obs = local_cond
        elif self.obs_as_global_cond:
            # condition throught global feature
            obs = global_cond.reshape(
                global_cond.shape[0], self.n_obs_steps, -1)
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
        ## -------------------- Projection Preparation END -------------------

        guide_step_cnt = 0
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # 2. predict model output
            model_output = model(
                trajectory, 
                t, 
                local_cond=local_cond, 
                global_cond=global_cond
            )
            # 3. compute previous timestep trj: x_t -> x_t-1
            scheduler_output_t = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            )

            ## --------------------------- Coupling ---------------------------
            trj_tm1 = scheduler_output_t.prev_sample

            # 4. check whether guidance is needed, if we use point estimate
            if self.pt_est and t >= max(1, self.t_stopgrad):
                with torch.no_grad():
                    # compute the original sample trj_0 from t_2: x_t-1 -> x_0
                    trj_0_from_tm1 = self.predict_x_0(
                        trj_tm1, 
                        t - 1, 
                        local_cond=local_cond, 
                        global_cond=global_cond,
                        scheduler=scheduler, 
                        generator=generator, 
                        **kwargs
                    )
                    cost = guide(
                        torch.tensor_split(trj_0_from_tm1, self.group_size, dim=0),
                    )
                    if cost.max() < threshold:
                        ## no guidance needed in this step
                        trajectory = trj_tm1
                        continue
                #end with [torch.no_grad()]
            #end if [t >= ...]

            # 5. perform gradient descent for guidance
            with torch.enable_grad():
                ## Get x_0_hat from timestep t: x_t -> x_0_hat
                trj_t = trajectory.detach().requires_grad_()
                if self.pt_est:
                    trj_0_hat_from_t = self.predict_x_0(
                        trj_t, 
                        t, 
                        local_cond=local_cond, 
                        global_cond=global_cond,
                        scheduler=scheduler, 
                        generator=generator, 
                        **kwargs
                    )

                    ## Compute the (common) cost for two players
                    cost, grads = guide.gradients(
                        input_vars = torch.tensor_split(trj_0_hat_from_t, self.group_size, dim=0),
                        with_respect_to = (trj_t,), 
                    )
                else:
                    ### Not using point estimate, so we compute cost directly from x_t 
                    cost, grads = guide.gradients(
                        input_vars = torch.tensor_split(trj_t, self.group_size, dim=0), 
                        with_respect_to = (trj_t,), 
                    )
                #end if [self.pt_est]
                grad = grads[0]  # gradients with respect to the trajectory

                ## prep the scaled gradient
                beta_t = scheduler.betas[t]       # scalar tensor
                alpha_t = scheduler.alphas[t]     # scalar tensor
                grad_scl = self.grad_scale * beta_t / torch.sqrt(alpha_t)
                scaled_grad = grad_scl * grad

                ## Do gradient descent for coupling via cost 
                for _ in range(self.n_guide_steps):
                    ## stop guidance near the end
                    if t < self.t_stopgrad:
                        break
                    
                    trj_tm1 = trj_tm1 - scaled_grad
                    guide_step_cnt += 1
                    if return_grads: 
                        guidance_gradients.append(grad.clone().detach().to('cpu'))
                #end for
            trajectory = trj_tm1
            ## -------------------------- Coupling END -----------------------------
        
            ## -------------------------- Projection -------------------------------
            # Project guided trajectory onto sets 
            h_start = self.n_obs_steps - int(self.oa_step_convention) 
            traj_to_proj = trajectory[..., h_start:, :]
            if self.timeit:
                raise NotImplementedError(
                    "Timeit is not supported in this policy yet.")
            else:
                projected_traj = self.projector.project(
                    traj_to_proj, 
                    params = proj_params, 
                )
            ## We put back the projected part without modifying the first n_obs_steps(-1) steps 
            ## which will be discarded later anyway 
            trajectory[..., h_start:, :] = projected_traj 
            ## Optionally, we can do inpainting on the first n_obs_steps(-1) steps 
            ## this might provide a "smoother" trajectory instead of one that is partially noisy and 
            ## partially projected ) 
            if h_start > 0:
                trajectory[..., :h_start, :] = nobs_in_nact[:, -h_start:,:]
            ## -------------------------- Projection END ---------------------------

            ## ----------------------- Log Intermediates -----------------------
            if intermediates_log is not None:
                step_data = {
                    't': int(t),
                    'x_tm1': trj_tm1.detach().cpu().clone(),
                    'x_before_proj': traj_to_proj.detach().cpu().clone(),
                    'x_after': trajectory.detach().cpu().clone(),
                }
                intermediates_log.append(step_data)
            ## ----------------------- Log Intermediates END -------------------
        #end for [t] loop
        
        # Check projected trajecoties
        # self.sanity_check(projected_traj, proj_params['pos_init'], proj_params['dx_max'], eps=1e-4)
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask] 

        # logger.info(f"Guided steps: {guide_step_cnt}\n")
        if return_grads:
            # guidance_gradients is a list of tensors, each tensor is of shape [B, H, D]
            # where B is the batch size, T is the number of timesteps, H is the horizon, and D is dimension
            # we stack them along the second dimension to get a tensor of shape [B, T_X, H, D]
            # if n_guide_steps == 1 and t_stopgrad == 1, then it will be of shape [B, T, H, D]
            guidance_gradients = torch.stack(guidance_gradients, dim=1) # [B, T, H, D] if n_guide_steps==1 && t_stopgrad==1

        if return_grads:
            return trajectory, guidance_gradients
        
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
        
    
    def predict_action(self, 
        obs_dict: Dict[str, torch.Tensor], 
        return_grads: bool = False,
    ) -> Dict[str, torch.Tensor]:
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
            return_grads=return_grads,
            **self.kwargs)
        if return_grads: 
            nsample, guidance_grads = nsample
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        if return_grads:
            # unnormalize gradients
            norm_scl:torch.Tensor = self.normalizer['action'].params_dict['scale'].data
            guidance_grads /= norm_scl.to(device=guidance_grads.device, dtype=guidance_grads.dtype)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
            if return_grads:
                guidance_grads = guidance_grads[:,:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if return_grads:
            result['guidance_grads'] = guidance_grads
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result
