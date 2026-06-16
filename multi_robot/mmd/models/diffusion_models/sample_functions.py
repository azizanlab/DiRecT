"""
Modified from 
- https://github.com/jacarvalho/mpd-public
- https://github.com/yoraish/mmd 
"""
import torch
from typing import Union, List, Tuple, Dict
from torch_robotics.torch_utils.torch_timer import TimerCUDA


from mmd.models.projection.projectors import BaseProjector


def apply_hard_conditioning_old(x, conditions):
    for t, val in conditions.items():
        x[:, t, :] = val.clone()
    # Set angle (dx dy) to be in the direction of the position finite difference.
    # dir_finite_diff = x[:, 1:, 0:2] - x[:, 0:-1, 0:2]
    # x[:, 1:, 2:4] = dir_finite_diff / torch.norm(dir_finite_diff, dim=-1, keepdim=True)
    return x

def apply_hard_conditioning(
    x:torch.Tensor, 
    conditions:Dict[Union[int, str], torch.Tensor]
) -> torch.Tensor:
    """
    Support groupwise hard conditioning, where the conditions are given in groups. 
    """
    if "group_size" not in conditions:
        # Fall back to the old version where all conditions are applied to all samples.
        for t, val in conditions.items():
            x[:, t, :] = val.clone()
        return x
    
    grp_sz = conditions.get("group_size", x.shape[0])
    grp_inds = torch.arange(x.shape[0], device=x.device).split(grp_sz)
    for t, val in conditions.items():
        if t == 'group_size':
            continue
        for g, b in enumerate(grp_inds):
            x[b, t, :] = val[:b.size(0), g, :].clone()  # hard_conds is in shape (B, G, D), where G is the group size.
    return x

def apply_cross_conditioning(x, conditions, transforms):
    for (m1, m2), (ind1, ind2) in conditions.items():
        transform_m1 = transforms[m1]
        transform_m2 = transforms[m2]
        relative_transform = transform_m2 - transform_m1
        if x[m1].shape[2] > relative_transform.shape[0]:
            relative_transform = torch.cat([relative_transform,
                                            torch.zeros(x[m1].shape[2] - relative_transform.shape[0],
                                                        device=relative_transform.device)])
        boundary = relative_transform / torch.norm(relative_transform, keepdim=True)
        # if element in boundary is 0, set it to be very big
        boundary[boundary == 0] = 1e6
        x[m1][:, ind1, :] = torch.min(x[m2][:, ind2, :] + relative_transform, boundary)
        x[m2][:, ind2, :] = torch.max(x[m1][:, ind1, :] - relative_transform, -boundary)
    return x


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def ddpm_sample_fn(
        model, x, hard_conds, context, t,
        guide=None,
        n_guide_steps=1,
        scale_grad_by_std=False,
        t_start_guide=torch.inf,
        noise_std_extra_schedule_fn=None,  # 'linear'
        **kwargs
):
    t_single = t[0]
    # By convention, a time less than zero gets zero noise. That is, sampling to update mean is computed to until t=0 and then only guidance and sampling, without updating the mean, is done.
    if t_single < 0:
        t = torch.zeros_like(t)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, hard_conds=hard_conds, context=context, t=t)
    x = model_mean

    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    if guide is not None and t_single < t_start_guide:
        x = guide_gradient_steps(
            x,
            hard_conds=hard_conds,
            guide=guide,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
            model_var=model_var,
        )

    noise = torch.randn_like(x)
    # No noise when t = 0.
    noise[t == 0] = 0

    # For smoother results, we can decay the noise standard deviation throughout the diffusion
    # this is roughly equivalent to using a temperature in the prior distribution
    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)

    values = None
    return x + model_std * noise * noise_std, values


def guide_gradient_steps(
        x,
        hard_conds=None,
        guide=None,
        n_guide_steps=1, scale_grad_by_std=False,
        model_var=None,
        **kwargs
):
    for _ in range(n_guide_steps):
        grad_scaled = guide(x, **kwargs)

        if scale_grad_by_std:
            grad_scaled = model_var * grad_scaled

        x = x + grad_scaled
        x = apply_hard_conditioning(x, hard_conds)

    return x


@torch.no_grad()
def projected_coupling_ddpm_sample_fn(
        model, x, hard_conds, context, t,
        guide=None,
        projector:BaseProjector=None,
        guide_kwargs={},
        project_params=None,
        n_guide_steps=1,
        scale_grad_by_std=False,
        t_start_guide=torch.inf,
        noise_std_extra_schedule_fn=None,  # 'linear'
        **kwargs
):
    t_single = t[0]
    # By convention, a time less than zero gets zero noise. That is, sampling to update mean is computed to until t=0 and then only guidance and sampling, without updating the mean, is done.
    if t_single < 0:
        raise ValueError(f"t must be non-negative for sampling. Got t = {t_single}.")
        t = torch.zeros_like(t)

    if t_single == 0:
        ## Not doing inpainting at t==0 to enforce projection constraints. 
        t_stps = []
        for k in hard_conds.keys():
            if isinstance(k, int) and k > 0 and k == x.shape[-2]-1:  # x[..., H, D]
                t_stps.append(k)
        max_t = max(t_stps) if t_stps else None
        if max_t is not None:
            max_t_hard_cond = {
                max_t: hard_conds.pop(max_t)
            }

    with TimerCUDA() as timer_diffuse: 
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, hard_conds=hard_conds, context=context, t=t)
    t_diffuse = timer_diffuse.elapsed

    x = model_mean
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    with TimerCUDA() as timer_coupling:
        if guide is not None and t_single < t_start_guide:
            x = guide_gradient_steps(
                x,
                hard_conds=hard_conds,
                guide=guide,
                n_guide_steps=n_guide_steps,
                scale_grad_by_std=scale_grad_by_std,
                model_var=model_var,
                **guide_kwargs
            )
    t_coupling = timer_coupling.elapsed

    noise = torch.randn_like(x)
    # No noise when t = 0.
    noise[t == 0] = 0

    # For smoother results, we can decay the noise standard deviation throughout the diffusion
    # this is roughly equivalent to using a temperature in the prior distribution
    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)

    x = x + model_std * noise * noise_std

    ##---------------------- Projector step ----------------------##
    t_project = 0.0
    x_before_proj = x.detach().cpu().clone() if kwargs.get('intermediates_log') is not None else None
    if projector is not None:
        assert project_params is not None, "Projector parameters must be provided."
        assert 'dx_max' in project_params, "Projector parameters must include 'dx_max'."
        assert 'pos_init' in project_params, "Projector parameters must include 'pos_init'."
        with TimerCUDA() as timer_project:
            x_1_on_pos = projector.project(
                x[:, 1:, :2],
                params=project_params,
            )
        t_project = timer_project.elapsed
        x[:, 1:, :2] = x_1_on_pos

    # Optionally log intermediates for visualization/debugging
    intermediates_log = kwargs.get('intermediates_log', None)
    if intermediates_log is not None:
        step_data = {
            't': int(t_single),
            'x': x.detach().cpu().clone(),
        }
        if projector is not None:
            step_data['x_before_proj'] = x_before_proj
            step_data['x_after_proj'] = x.detach().cpu().clone()
        intermediates_log.append(step_data)

    values = None
    time_elapsed = {
        'diffuse': t_diffuse,
        'coupling': t_coupling,
        'project': t_project,
    }

    if 'timeit' in kwargs and kwargs['timeit']:
        return x, values, time_elapsed
    else:
        return x, values
