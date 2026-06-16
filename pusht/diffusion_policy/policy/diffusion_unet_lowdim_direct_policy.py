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

class DiffusionUnetLowdimDiRecTPolicy(DiffusionUnetLowdimCostPolicy):
    def __init__(self,
                guide: CostGuide,
                projector: BaseProjectionOperator,
                project_params: Union[Dict, omegaconf.dictconfig.DictConfig],
                *args,
                n_guide_steps=1,
                grad_scale=1.,
                t_stopgrad = 2,
                t_start_guide = None,
                t_start_projection = None,
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
        # t_start_guide: guidance only when t < t_start_guide. None = all timesteps.
        # guidance active when t_stopgrad <= t < t_start_guide.
        self.t_start_guide = t_start_guide
        # t_start_projection: projection only when t < t_start_projection. None = all timesteps.
        self.t_start_projection = t_start_projection

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
        return self._conditional_sample_pgd(
            condition_data, condition_mask,
            guide=guide,
            local_cond=local_cond, global_cond=global_cond,
            generator=generator, **kwargs)

    def _conditional_sample_pgd(self,
            condition_data, condition_mask,
            guide=None,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
    )->torch.Tensor:
        """
        PGD mode: one (or more) iterations of Projected Gradient Descent on
            min_x  C(x) + α_s² ||x - x̂_0||²   s.t. h(x) ≤ 0

        Each PGD iteration:
          1. x = x - η * (∇C(x) + 2α_s²(x - x̂_0))
          2. x = Π_K(x)   [project onto constraint set]

        Both guidance and projection operate in clean space via Tweedie.
        The proximity term α_s² ||x - x̂_0||² penalizes deviation from the
        diffusion model's prediction, with α_s naturally scaling trust.
        """
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

        scheduler.set_timesteps(self.num_inference_steps)
        if return_grads:
            guidance_gradients:List[torch.Tensor] = []

        ## -------------------- Projection Preparation -------------------
        if self.obs_as_local_cond:
            obs = local_cond
        elif self.obs_as_global_cond:
            obs = global_cond.reshape(
                global_cond.shape[0], self.n_obs_steps, -1)
        else:
            obs = condition_data[condition_mask]
        assert obs.ndim == 3

        obs_unnormed = self.normalizer['obs'].unnormalize(obs.detach().clone())
        nobs_in_nact = self.normalizer['action'].normalize(
            obs_unnormed[:, :, -self.action_dim:]
        )
        proj_params = {
            'pos_init': nobs_in_nact[:, -1, :],
            'dx_max': self.project_params['vel_max'] * self.project_params['delta_t'],
        }
        h_start = self.n_obs_steps - int(self.oa_step_convention)
        ## -------------------- Projection Preparation END -------------------

        guide_step_cnt = 0
        t_start_guide = self.t_start_guide
        t_start_projection = self.t_start_projection
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. DDPM step: x_t -> x_{t-1}
            model_output = model(
                trajectory, t,
                local_cond=local_cond,
                global_cond=global_cond
            )
            scheduler_output_t = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs
            )
            trajectory = scheduler_output_t.prev_sample

            ## ---- Determine what to apply this step ----
            apply_guide = (t >= self.t_stopgrad) and \
                          (t_start_guide is None or t < t_start_guide)
            apply_proj = (t_start_projection is None or t < t_start_projection)

            if apply_guide or apply_proj:
                ## ---- Tweedie estimate ----
                if t == 0:
                    x_hat_0 = trajectory.clone()
                    alpha_s_sq = 1.0
                    g_sq = 1.0
                else:
                    with torch.no_grad():
                        x_hat_0 = self.predict_x_0(
                            trajectory, t - 1,
                            local_cond=local_cond, global_cond=global_cond,
                            scheduler=scheduler, generator=generator, **kwargs
                        )
                    alpha_s_sq = scheduler.alphas_cumprod[t - 1].item()
                    g_sq = scheduler.betas[t - 1].item()

                x_hat_0_original = x_hat_0[..., h_start:, :].clone()

                ## ---- PGD iterations ----
                if apply_guide and self.n_guide_steps > 0:
                    # Initialize x at x̂_0
                    x_action = x_hat_0[..., h_start:, :].detach().clone()

                    for pgd_iter in range(self.n_guide_steps):
                        # Compute cost gradient ∇C(x)
                        with torch.enable_grad():
                            x_action_var = x_action.detach().requires_grad_()
                            cost, grads = guide.gradients(
                                input_vars=torch.tensor_split(x_action_var, self.group_size, dim=0),
                                with_respect_to=(x_action_var,),
                            )
                            cost_grad = grads[0].detach()

                        # Proximity gradient: 2(α_s²/g²)(x - x̂_0)
                        prox_grad = 2.0 * (alpha_s_sq / g_sq) * (x_action - x_hat_0_original)

                        # PGD gradient step
                        x_action = x_action - self.grad_scale * (cost_grad + prox_grad)

                        guide_step_cnt += 1
                        if return_grads:
                            guidance_gradients.append(cost_grad.clone().to('cpu'))

                        # Project onto constraint set (inner projection per PGD iteration)
                        if apply_proj:
                            if h_start > 0:
                                x_hat_0_temp = x_hat_0.clone()
                                x_hat_0_temp[..., h_start:, :] = x_action
                                x_hat_0_temp[..., :h_start, :] = nobs_in_nact[:, -h_start:, :]
                            with torch.no_grad():
                                x_action = self.projector.project(
                                    x_action, params=proj_params)

                    x_hat_0 = x_hat_0.clone()
                    x_hat_0[..., h_start:, :] = x_action.detach()

                elif apply_proj:
                    ## Projection only (no guidance)
                    with torch.no_grad():
                        x_hat_0_proj = self.projector.project(
                            x_hat_0[..., h_start:, :], params=proj_params)
                        x_hat_0[..., h_start:, :] = x_hat_0_proj

                ## ---- Map back to noisy space ----
                with torch.no_grad():
                    if t == 0:
                        trajectory[..., h_start:, :] = x_hat_0[..., h_start:, :]
                    else:
                        alpha_s = scheduler.alphas_cumprod[t - 1].sqrt()
                        trajectory[..., h_start:, :] += alpha_s * (
                            x_hat_0[..., h_start:, :] - x_hat_0_original)

                    # Inpaint observation steps
                    if h_start > 0:
                        trajectory[..., :h_start, :] = nobs_in_nact[:, -h_start:, :]

            ## ----------------------- Log Intermediates -----------------------
            if intermediates_log is not None:
                step_data = {
                    't': int(t),
                    'x_tm1': scheduler_output_t.prev_sample.detach().cpu().clone(),
                    'x_after': trajectory.detach().cpu().clone(),
                }
                if (apply_guide or apply_proj) and t > 0:
                    step_data['x_hat_0'] = x_hat_0.detach().cpu().clone()
                intermediates_log.append(step_data)
            ## ----------------------- Log Intermediates END -------------------
        #end for [t] loop

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        if return_grads:
            guidance_gradients = torch.stack(guidance_gradients, dim=1)
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
