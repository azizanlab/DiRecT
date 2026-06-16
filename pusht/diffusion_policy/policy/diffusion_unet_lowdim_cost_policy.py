from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import logging
from einops import rearrange, reduce
from copy import deepcopy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.guide.cost_guide import CostGuide
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

logger = logging.getLogger(__name__)

class DiffusionUnetLowdimCostPolicy(DiffusionUnetLowdimPolicy):
    def __init__(self,
                guide: CostGuide,
                n_guide_steps=2,
                grad_scale=1.,
                t_stopgrad = 2,
                threshold=1e-6,
                point_estimate=False,
                group_size=2,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)

        self.guide = guide
        self.n_guide_steps = n_guide_steps
        self.grad_scale = grad_scale
        self.t_stopgrad = t_stopgrad
        self.threshold = threshold
        self.pt_est:bool = point_estimate
        self.group_size = group_size
    
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            guide=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        if guide is None:
            guide = self.guide
        threshold = self.threshold
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

                ## Do gradient descent for coupling via cost 
                for _ in range(self.n_guide_steps):
                    ## stop guidance near the end
                    if t < self.t_stopgrad:
                        break
                    
                    trj_tm1 = trj_tm1 - self.grad_scale * grad
                    guide_step_cnt += 1
                    
                #end for
            trajectory = trj_tm1

            ## ----------------------- Log Intermediates -----------------------
            if intermediates_log is not None:
                step_data = {
                    't': int(t),
                    'x_tm1': trj_tm1.detach().cpu().clone(),
                    'x_after': trajectory.detach().cpu().clone(),
                }
                intermediates_log.append(step_data)
            ## ----------------------- Log Intermediates END -------------------
        #end for

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask] 

        # logger.info(f"Guided steps: {guide_step_cnt}\n")

        return trajectory

    
    def predict_x_0(self, 
                    x_t, 
                    t,
                    local_cond, global_cond,
                    scheduler: DDPMScheduler, 
                    generator, 
                    **kwargs):
        model = self.model
        model_output_t = model(
            x_t, 
            t, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        x_0 = scheduler.step(
            model_output_t, 
            t, 
            x_t, 
            generator=generator,
            **kwargs
        ).pred_original_sample

        return x_0
