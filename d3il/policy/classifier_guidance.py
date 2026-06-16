"""
ClassifierGuidancePolicy: standard classifier guidance during DDPM reverse process.

Evaluates the value function directly on the noisy x_t and shifts the DDPM
posterior mean by guidance_lr * posterior_var * grad(value(x_t)).
Single gradient evaluation per step (no inner loop).
Fully batched on GPU.
"""

import math
import time
from collections import namedtuple

import numpy as np
import torch
from torch import nn

from model import CosineScheduleDiffusion, apply_conditioning
from utils.arrays import to_np, to_torch, apply_dict

Trajectories = namedtuple("Trajectories", "actions observations values")


def _zero_conditioned_grad(g, conditions, action_dim):
    """Zero out gradient at conditioned observation positions."""
    for t, val in conditions.items():
        g[:, t, action_dim:] = 0.0
    return g


class ClassifierGuidancePolicy(nn.Module):
    """DDPM sampling with classifier guidance.

    Reverse-diffuses from t=1 to t=0, shifting the posterior mean by
    the gradient of the value function evaluated on the noisy x_t,
    scaled by the posterior variance.
    """

    def __init__(
        self,
        model,
        normalizer,
        value_model,
        action_dim,
        state_dim,
        horizon,
        n_sampling_steps=10,
        guidance_lr=0.05,
        gradient_start=0.0,
        device="cuda:0",
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.value_model = value_model
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_sampling_steps = n_sampling_steps
        self.guidance_lr = guidance_lr
        self.gradient_start = gradient_start
        self.device = device

        self.diffusion = CosineScheduleDiffusion(model, action_dim)

    def __call__(self, conditions, batch_size=1):
        """Sample trajectories with classifier guidance.

        Same interface as UnconditionalPolicy.__call__.
        """
        conditions = apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        t_start = time.time()

        transition_dim = self.action_dim + self.state_dim
        conditions_torch = to_torch(conditions, device=self.device)

        # start from pure noise
        x = torch.randn(
            batch_size, self.horizon, transition_dim, device=self.device
        )
        x = apply_conditioning(x, conditions_torch, self.action_dim)

        # time grid: t=1 (noisy) to t=0 (clean)
        timesteps = torch.linspace(1.0, 0.0, self.n_sampling_steps + 1, device=self.device)

        save_chain = (batch_size == 1)
        if save_chain:
            denoising_chain = [x.clone()]
            x0_estimates = []

        for i in range(self.n_sampling_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            is_last_step = (i == self.n_sampling_steps - 1)

            # --- standard DDPM x0 prediction ---
            abar_now = self.diffusion.alpha_bar(t_now)
            abar_next = self.diffusion.alpha_bar(t_next)

            t_batch = t_now.expand(batch_size)
            x0_pred = self.model(x, t_batch)
            x0_pred = apply_conditioning(x0_pred, conditions_torch, self.action_dim)

            if save_chain:
                x0_estimates.append(x0_pred.detach().clone())

            # --- DDPM posterior ---
            beta = 1.0 - abar_now / abar_next
            coeff_x0 = math.sqrt(abar_next) * beta / (1.0 - abar_now)
            coeff_xt = math.sqrt(1.0 - beta) * (1.0 - abar_next) / (1.0 - abar_now)
            posterior_mean = coeff_x0 * x0_pred + coeff_xt * x
            posterior_var = max((1.0 - abar_next) / (1.0 - abar_now) * beta, 0.0)

            # --- classifier guidance: shift posterior mean ---
            frac_done = i / self.n_sampling_steps
            if frac_done >= self.gradient_start and posterior_var > 0:
                x_in = x.detach().clone().requires_grad_(True)
                value = self.value_model(x_in)  # evaluate on noisy x_t
                loss = -value.sum()
                g = torch.autograd.grad(loss, x_in)[0]
                g = _zero_conditioned_grad(g, conditions_torch, self.action_dim)
                posterior_mean = posterior_mean + self.guidance_lr * posterior_var * g

            # --- sample ---
            if is_last_step:
                x = posterior_mean
            else:
                noise = torch.randn_like(x)
                x = posterior_mean + math.sqrt(posterior_var) * noise

            x = apply_conditioning(x, conditions_torch, self.action_dim)
            x = x.detach()

            if save_chain:
                denoising_chain.append(x.clone())

        # unnormalize
        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(to_np(normed_actions), "actions")

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(to_np(normed_observations), "observations")

        trajectories = Trajectories(actions, observations, np.zeros((batch_size, 1)))

        t_end = time.time()
        info = {"computation_time": t_end - t_start}

        if save_chain:
            chain = torch.stack(denoising_chain, dim=1)
            chain = self._unnormalize_chain(chain)
            x0_est = torch.stack(x0_estimates, dim=1)
            x0_est = self._unnormalize_chain(x0_est)
            return actions[:, 0], trajectories, chain, x0_est, info
        else:
            return actions[:, 0], trajectories, None, None, info

    def _unnormalize_chain(self, chain):
        """Unnormalize a chain of shape (batch, steps, horizon, transition_dim)."""
        obs_chain = chain[..., self.action_dim:]
        obs_unnorm = self.normalizer.unnormalize(to_np(obs_chain), "observations")
        act_chain = chain[..., :self.action_dim]
        act_unnorm = self.normalizer.unnormalize(to_np(act_chain), "actions")
        return np.concatenate([act_unnorm, obs_unnorm], axis=-1)
