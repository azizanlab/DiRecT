import copy
from typing import Dict, Any, Tuple, Optional, Callable, List
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
import torch
import lightning.pytorch as pl

import casadi as cs
from qpth.qp import QPFunction, QPSolvers

from env import Maze2dConstrainedEnv
from diffusion import Diffusion, apply_conditioning

IPOPT_VERBOSE_LEVEL = 1


## Utils
def apply_dict(d: Dict[Any, Any], f: Callable):
    for k, v in d.items():
        if isinstance(v, dict):
            apply_dict(v, f)
            return
        d[k] = f(v)


def to_np(x: Any) -> NDArray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


# Remove constraints that do not depend on optimization vars from optimization
def vec_dep(gvec, x):
    gvec = cs.vec(gvec)
    xnames = {v.name() for v in cs.symvar(cs.vec(x))}
    idx = [
        i for i in range(int(gvec.size1())) if any(v.name() in xnames for v in cs.symvar(gvec[i]))
    ]
    return cs.vertcat(*[gvec[i] for i in idx]) if idx else cs.MX.zeros(0, 1)  # type: ignore


# Flatten the condition dict into a list
def flatten_conditions(conditions: Dict[Any, Any]) -> List[Any]:
    flattened_conditions = []
    for i in sorted(conditions.keys()):
        if not isinstance(i, int):
            continue
        flattened_conditions.append(conditions[i])
    return flattened_conditions


# [Note]: we use eta=1 which yields Markovian DDPM equivalent transitions
class DDIMSampler:
    """DDIM sampler with stochastic interpolation.

    Args:
        model: Trained diffusion model.
        interpolation_coeff: eta parameter. 0 = deterministic DDIM, 1 = stochastic DDPM.
    """

    def __init__(self, model: Diffusion, interpolation_coeff: float = 0.0, **kwargs):
        del kwargs
        self.model = model
        self.interpolation_coeff = interpolation_coeff

    def ddim_sample(
        self,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        epsilon: torch.Tensor,
        conditions: Dict[Any, Any],
    ):
        eta = self.interpolation_coeff
        alpha_t = self.model.noise_schedule.alpha_t(t)
        alpha_s = self.model.noise_schedule.alpha_t(s)
        sigma_t = self.model.noise_schedule.sigma_t(t)
        sigma_s = self.model.noise_schedule.sigma_t(s)
        alpha_ts = alpha_t / alpha_s
        sigma_ts = torch.sqrt(sigma_t**2 - (alpha_ts * sigma_s) ** 2)
        c_st = eta * sigma_s / sigma_t * sigma_ts
        mu = alpha_s * x1 + (sigma_s**2 - c_st**2).clamp(min=0.0).sqrt() / sigma_t * (
            xt - alpha_t * x1
        )
        std = c_st
        sample = mu + std * epsilon
        sample = apply_conditioning(
            x=sample, conditions=conditions, action_dim=2, zero_conditioning=False
        )
        return sample

    def integrate_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        model_out: torch.Tensor,
        conditions: Dict[Any, Any] = {},
        epsilon: Optional[torch.Tensor] = None,
        score_correction: Optional[torch.Tensor] = None,
        last_step: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if epsilon is None:
            epsilon = torch.randn_like(x)
        x1_estimate = self.model.estimate_data(model_out, x, t)
        if score_correction is not None:
            # Correction by Tweedie's formula
            alpha_t = self.model.noise_schedule.alpha_t(t)
            sigma_t = self.model.noise_schedule.sigma_t(t)
            x1_estimate = x1_estimate + sigma_t**2 / alpha_t.clamp(min=1e-8) * score_correction
        if last_step:
            return x1_estimate
        xn = self.ddim_sample(x, x1_estimate, t, s, epsilon, conditions)
        return xn


class Policy(pl.LightningModule):
    """Base class for all sampling policies.

    Args:
        env: Constrained maze environment with obstacle and constraint definitions.
        diffuser: Trained diffusion model.
        sampling_steps: Number of denoising steps.
        sampler_interpolation_coeff: DDIM eta (0=deterministic, 1=stochastic).
        time_discretization: Timestep spacing ("uniform", "log-noise", or "log-snr").
        time_discretization_kwargs: Extra params (e.g. prior_time_offset).
        policy_kwargs: Algorithm-specific parameters passed to subclass. May include
            "ipopt_options": a dict of solver options (e.g. {"ipopt.max_iter": 200})
            merged over the defaults for all policies that solve NLPs with IPOPT.
    """

    def __init__(
        self,
        env: Maze2dConstrainedEnv,
        diffuser: Diffusion,
        sampling_steps: int,
        sampler_interpolation_coeff: float = 0.0,
        time_discretization: str = "uniform",
        time_discretization_kwargs: Dict[str, Any] = {},
        policy_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.diffuser = diffuser
        self.env = env
        self.noise_schedule = self.diffuser.noise_schedule
        self.time_discretization = time_discretization
        self.sampling_steps = sampling_steps
        self.action_dim = self.env.action_dim
        self.state_dim = self.env.state_dim
        self.transition_dim = self.env.transition_dim

        assert self.time_discretization in {"uniform", "log-noise", "log-snr"}

        # Choose timestep discretization function
        match time_discretization:
            case "uniform":
                prior_time_offset = time_discretization_kwargs.get("prior_time_offset", 0.0)

                def get_timesteps(num_steps: int) -> torch.Tensor:
                    return torch.linspace(
                        start=prior_time_offset,
                        end=1.0,
                        steps=num_steps + 1,
                        device=self.device,
                        dtype=self.dtype,  # type: ignore
                    )

                self.get_timesteps = get_timesteps
            case "log-noise":
                sigma_min = time_discretization_kwargs.get("sigma_min", 0.001)
                sigma_max = time_discretization_kwargs.get("sigma_max", 0.999)
                schedule_sigma_min = self.noise_schedule.sigma_t(torch.tensor(1.0)).nan_to_num(
                    nan=0.0
                )
                schedule_sigma_max = self.noise_schedule.sigma_t(torch.tensor(0.0)).nan_to_num(
                    nan=torch.inf
                )
                log_sigma_min = (
                    torch.maximum(
                        schedule_sigma_min,
                        torch.as_tensor(sigma_min),
                    )
                    .log()
                    .item()
                )
                log_sigma_max = (
                    torch.minimum(
                        schedule_sigma_max,
                        torch.as_tensor(sigma_max),
                    )
                    .log()
                    .item()
                )

                def get_timesteps(num_steps: int) -> torch.Tensor:
                    sigmas = torch.linspace(
                        start=log_sigma_max,
                        end=log_sigma_min,
                        steps=num_steps + 1,
                        device=self.device,
                        dtype=self.dtype,  # type: ignore
                    ).exp()
                    return self.noise_schedule.inverse_sigma(sigmas)

                self.get_timesteps = get_timesteps
            case "log-snr":
                log_snr_min = time_discretization_kwargs.get("log_snr_min", -15.0)
                log_snr_max = time_discretization_kwargs.get("log_snr_max", 15.0)
                schedule_log_snr_max = (
                    (
                        self.noise_schedule.alpha_t(torch.tensor(1.0))
                        / self.noise_schedule.sigma_t(torch.tensor(1.0)) ** 2
                    )
                    .log()
                    .nan_to_num(nan=torch.inf)
                )
                schedule_log_snr_min = (
                    (
                        self.noise_schedule.alpha_t(torch.tensor(0.0))
                        / self.noise_schedule.sigma_t(torch.tensor(0.0)) ** 2
                    )
                    .log()
                    .nan_to_num(nan=-torch.inf)
                )
                log_snr_min = torch.maximum(
                    schedule_log_snr_min,
                    torch.as_tensor(log_snr_min),
                ).item()
                log_snr_max = torch.minimum(
                    schedule_log_snr_max, torch.as_tensor(log_snr_max)
                ).item()

                def get_timesteps(num_steps: int) -> torch.Tensor:
                    log_snr = torch.linspace(
                        start=log_snr_min,
                        end=log_snr_max,
                        steps=num_steps + 1,
                        device=self.device,
                        dtype=self.dtype,  # type: ignore
                    )
                    return self.noise_schedule.inverse_log_snr(log_snr)

                self.get_timesteps = get_timesteps

        ## Sampler
        self.sampler = DDIMSampler(model=diffuser, interpolation_coeff=sampler_interpolation_coeff)

        ## IPOPT solver options (user overrides take precedence over defaults)
        self.ipopt_options = {
            "ipopt.print_level": IPOPT_VERBOSE_LEVEL,
            **policy_kwargs.get("ipopt_options", {}),
        }

    def preprocess_conditions(self, conditions: Dict[Any, Any]) -> Dict[Any, Any]:
        normed_conditions = copy.deepcopy(conditions)
        apply_dict(
            normed_conditions,
            lambda x: torch.as_tensor(
                self.env.normalizer.normalize(to_np(x), key="observations"),
                dtype=x.dtype,
                device=x.device,
            ),
        )
        return normed_conditions

    def post_process_trajectory_data(self, trajectory_data: Dict[str, Any]):
        # Observations
        assert "observation" in trajectory_data
        observations = trajectory_data["observation"]
        observations = torch.as_tensor(
            self.env.normalizer.unnormalize(to_np(observations), "observations"),
            device=observations.device,
            dtype=observations.dtype,
        )
        trajectory_data["observation"] = observations

        # Actions
        assert "actions" in trajectory_data
        actions = trajectory_data["actions"]
        actions = torch.as_tensor(
            self.env.normalizer.unnormalize(to_np(actions), "actions"),
            device=actions.device,
            dtype=actions.dtype,
        )
        trajectory_data["actions"] = actions

        # Intermediates
        trajectory_data["intermediates"] = self.env.unnormalize_chain(
            trajectory_data["intermediates"]
        )

        # Data estimates
        trajectory_data["data_estimates"] = self.env.unnormalize_chain(
            trajectory_data["data_estimates"]
        )

        # Solver statistics (only present for policies that ran IPOPT solves)
        self.add_ipopt_stats(trajectory_data)

    def reset_ipopt_stats(self):
        self._ipopt_solves = 0
        self._ipopt_iters = 0
        self._ipopt_failures = 0

    def record_ipopt_solve(self, stats: Optional[Dict[str, Any]], failed: bool = False):
        """Accumulate per-solve IPOPT statistics (iteration count and convergence)."""
        self._ipopt_solves += 1
        if stats is not None:
            self._ipopt_iters += int(stats.get("iter_count", 0))
            failed = failed or not stats.get("success", False)
        self._ipopt_failures += int(failed)

    def add_ipopt_stats(self, result_dict: Dict[str, Any]):
        if getattr(self, "_ipopt_solves", 0) > 0:
            result_dict["avg_ipopt_iters"] = self._ipopt_iters / self._ipopt_solves
            result_dict["ipopt_failure_rate"] = self._ipopt_failures / self._ipopt_solves

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], **kwargs) -> Dict[str, Any]:
        del conditions, kwargs
        raise NotImplementedError

    @torch.no_grad()
    def warmstart(self, conditions: Dict[Any, Any], batch_size: int) -> torch.Tensor:
        timesteps = self.get_timesteps(self.sampling_steps)

        intermediates = []

        x = torch.randn(
            batch_size,
            self.diffuser.horizon,
            self.transition_dim,
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for i in range(self.sampling_steps):
            model_out = self.diffuser.forward(x, timesteps[i], conditions=conditions)
            x = self.sampler.integrate_step(
                x,
                timesteps[i],
                timesteps[i + 1],
                model_out,
                conditions,
                last_step=i == (self.sampling_steps - 1),
            )
            intermediates.append(x.detach().clone())

        # Compute proxy env value
        value = self.env.proxy_value(x)

        # Sort and Select based on best proxy values
        sorted_indices = value.argsort(dim=0, descending=True).flatten().clone()
        value = value[sorted_indices, ...]
        x = x[sorted_indices, ...]
        intermediates = torch.stack(intermediates, dim=1)[sorted_indices, ...]
        return intermediates[0]


# Equivalent to Diffuser: https://github.com/jannerm/diffuser
# [Note]: Time is transformed from discrete [256 -> 1] to continuous [0 -> 1]
class NoGuidancePolicy(Policy):
    """Unconstrained diffusion sampling. No guidance applied."""
    def __init__(
        self,
        env: Maze2dConstrainedEnv,
        diffuser: Diffusion,
        sampling_steps: int,
        sampler_interpolation_coeff: float = 0.0,
        time_discretization: str = "uniform",
        time_discretization_kwargs: Dict[str, Any] = {},
        policy_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(
            env,
            diffuser,
            sampling_steps,
            sampler_interpolation_coeff,
            time_discretization,
            time_discretization_kwargs,
            policy_kwargs,
        )

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], batch_size=1, **kwargs) -> Dict[str, Any]:
        del kwargs

        result_dict = {}
        timesteps = self.get_timesteps(self.sampling_steps)
        conditions = self.preprocess_conditions(conditions)

        # START
        intermediates = []
        data_estimates = []
        start_time = perf_counter()

        x = torch.randn(
            batch_size,
            self.diffuser.horizon,
            self.transition_dim,  # [Transition-dim]
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for i in range(self.sampling_steps):
            model_out = self.diffuser.forward(x, timesteps[i], conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, timesteps[i])
            x = self.sampler.integrate_step(
                x,
                timesteps[i],
                timesteps[i + 1],
                model_out,
                conditions,
                last_step=i == (self.sampling_steps - 1),
            )
            intermediates.append(x.detach().clone())
            data_estimates.append(x1.detach().clone())
        data_estimates.append(x.detach().clone())

        # Compute proxy env value
        value = self.env.proxy_value(x)

        # END
        end_time = perf_counter()
        computation_time = end_time - start_time
        result_dict.update(
            {
                "computation_time": computation_time,
                "observation": x[:, :, self.action_dim :].clone(),
                "actions": x[:, :, : self.action_dim].clone(),
                "intermediates": torch.stack(intermediates, dim=1),
                "data_estimates": torch.stack(data_estimates, dim=1),
                "control_norm": torch.zeros_like(value),
                "proxy_value": value,
            }
        )
        self.post_process_trajectory_data(trajectory_data=result_dict)
        return result_dict


# [Note]: Time is transformed from discrete [256 -> 1] to continuous [0 -> 1]
class GradientGuidancePolicy(Policy):
    """Gradient guidance on the clean data estimate x1 = E[x_1 | x_t].

    Computes grad of the proxy value w.r.t. x_t through the data estimate,
    and applies gradient descent before the sampler step.

    Args (policy_kwargs):
        guidance_start_fraction: Fraction of steps before guidance begins.
        guidance_iterations: Number of gradient steps per denoising step.
        guidance_lr: Step size for gradient update.
    """

    def __init__(
        self,
        env: Maze2dConstrainedEnv,
        diffuser: Diffusion,
        sampling_steps: int,
        sampler_interpolation_coeff: float = 0.0,
        time_discretization: str = "uniform",
        time_discretization_kwargs: Dict[str, Any] = {},
        policy_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(
            env,
            diffuser,
            sampling_steps,
            sampler_interpolation_coeff,
            time_discretization,
            time_discretization_kwargs,
            policy_kwargs,
        )
        self.guidance_iterations = policy_kwargs["guidance_iterations"]
        self.guidance_lr = policy_kwargs["guidance_lr"]
        self.guidance_start_fraction = policy_kwargs["guidance_start_fraction"]

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], batch_size=1, **kwargs) -> Dict[str, Any]:
        del kwargs
        result_dict = {}
        timesteps = self.get_timesteps(self.sampling_steps)
        conditions = self.preprocess_conditions(conditions)

        # START
        intermediates = []
        data_estimates = []
        start_time = perf_counter()

        x = torch.randn(
            batch_size,
            self.diffuser.horizon,
            self.transition_dim,  # [Transition-dim]
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )
        total_control = 0.0
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for i in range(self.sampling_steps):
            t = timesteps[i]
            s = timesteps[i + 1]

            # reference to compute control norm
            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)

            epsilon = torch.randn_like(x)
            x_uncontrolled = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )

            apply_control = (i / self.sampling_steps) >= self.guidance_start_fraction
            if apply_control:
                for _ in range(self.guidance_iterations):
                    with torch.enable_grad():
                        inputs = x.detach().clone().requires_grad_(True)
                        model_out = self.diffuser.forward(inputs, t, conditions=conditions)
                        data_estimate = self.diffuser.estimate_data(model_out, inputs, t)
                        loss = -self.env.proxy_value(data_estimate).sum()
                        (g,) = torch.autograd.grad(loss, inputs, create_graph=False)
                    g = apply_conditioning(
                        g.detach(),
                        conditions=conditions,
                        action_dim=self.action_dim,
                        zero_conditioning=True,
                    )
                    x = x - self.guidance_lr * g
                    x = x.detach()

            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)
            x = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )
            intermediates.append(x.detach().clone())
            data_estimates.append(x1.detach().clone())

            # Compute control
            # x - x_uncontrolled = u Delta T
            # total_norm = sum( ||u||*2 Delta T ) = sum(||x - x_uncontrolled||^2 / Delta T)
            control = (x - x_uncontrolled).square().sum()
            total_control += control / (s - t)

        data_estimates.append(x.detach().clone())

        # Compute proxy env value
        value = self.env.proxy_value(x)

        # END
        end_time = perf_counter()
        computation_time = end_time - start_time
        result_dict.update(
            {
                "computation_time": computation_time,
                "observation": x[:, :, self.action_dim :].clone(),
                "actions": x[:, :, : self.action_dim].clone(),
                "intermediates": torch.stack(intermediates, dim=1),
                "data_estimates": torch.stack(data_estimates, dim=1),
                "proxy_value": value,
                "total_control": total_control,
            }
        )
        self.post_process_trajectory_data(trajectory_data=result_dict)
        return result_dict


# Inspired from Posterior Sampling methods: https://github.com/dps2022/diffusion-posterior-sampling
class ClassifierGuidancePolicy(Policy):
    """Classifier guidance directly on the noisy trajectory x_t.

    Unlike GradientGuidancePolicy which guides through the clean data estimate,
    this computes the gradient of the proxy value directly on x_t.

    Args (policy_kwargs):
        guidance_start_fraction: Fraction of steps before guidance begins.
        guidance_iterations: Number of gradient steps per denoising step.
        guidance_lr: Step size for gradient update.
    """

    def __init__(
        self,
        env: Maze2dConstrainedEnv,
        diffuser: Diffusion,
        sampling_steps: int,
        sampler_interpolation_coeff: float = 0.0,
        time_discretization: str = "uniform",
        time_discretization_kwargs: Dict[str, Any] = {},
        policy_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(
            env,
            diffuser,
            sampling_steps,
            sampler_interpolation_coeff,
            time_discretization,
            time_discretization_kwargs,
            policy_kwargs,
        )
        self.guidance_iterations = policy_kwargs["guidance_iterations"]
        self.guidance_lr = policy_kwargs["guidance_lr"]
        self.guidance_start_fraction = policy_kwargs["guidance_start_fraction"]

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], batch_size=1, **kwargs) -> Dict[str, Any]:
        del kwargs
        result_dict = {}
        timesteps = self.get_timesteps(self.sampling_steps)
        conditions = self.preprocess_conditions(conditions)

        # START
        intermediates = []
        data_estimates = []
        start_time = perf_counter()

        x = torch.randn(
            batch_size,
            self.diffuser.horizon,
            self.transition_dim,  # [Transition-dim]
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )
        total_control = 0.0
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for i in range(self.sampling_steps):
            t = timesteps[i]
            s = timesteps[i + 1]

            # reference to compute control norm
            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)

            epsilon = torch.randn_like(x)
            x_uncontrolled = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )

            apply_control = (i / self.sampling_steps) >= self.guidance_start_fraction
            if apply_control:
                for _ in range(self.guidance_iterations):
                    with torch.enable_grad():
                        inputs = x.detach().clone().requires_grad_(True)
                        loss = -self.env.proxy_value(inputs).sum()
                        (g,) = torch.autograd.grad(loss, inputs, create_graph=False)
                    g = apply_conditioning(
                        g.detach(),
                        conditions=conditions,
                        action_dim=self.action_dim,
                        zero_conditioning=True,
                    )
                    x = x - self.guidance_lr * g
                    x = x.detach()

            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)
            x = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )
            intermediates.append(x.detach().clone())
            data_estimates.append(x1.detach().clone())

            # Compute control
            # x - x_uncontrolled = u Delta T
            # total_norm = sum( ||u||*2 Delta T ) = sum(||x - x_uncontrolled||^2 / Delta T)
            control = (x - x_uncontrolled).square().sum()
            total_control += control / (s - t)

        data_estimates.append(x.detach().clone())

        # Compute proxy env value
        value = self.env.proxy_value(x)

        # END
        end_time = perf_counter()
        computation_time = end_time - start_time
        result_dict.update(
            {
                "computation_time": computation_time,
                "observation": x[:, :, self.action_dim :].clone(),
                "actions": x[:, :, : self.action_dim].clone(),
                "intermediates": torch.stack(intermediates, dim=1),
                "data_estimates": torch.stack(data_estimates, dim=1),
                "proxy_value": value,
                "total_control": total_control,
            }
        )
        self.post_process_trajectory_data(trajectory_data=result_dict)
        return result_dict


# Inspired by PDM: https://github.com/RAISELab-atUVA/Projected-Diffusion
class ProjectionPolicy(Policy):
    """Project each denoising step onto the feasible set via IPOPT.

    Solves min ||x - x_ref||^2 s.t. constraints after the sampler step.
    Optionally combines with gradient guidance before projection.

    Args (policy_kwargs):
        guidance_start_fraction: Fraction of steps before gradient guidance begins.
        guidance_iterations: Number of gradient steps (0 to disable).
        guidance_lr: Step size for gradient update.
        projection_start_fraction: Fraction of steps before projection begins.
        ipopt_options: Optional dict of IPOPT solver options merged over defaults.
    """

    def __init__(
        self,
        env: Maze2dConstrainedEnv,
        diffuser: Diffusion,
        sampling_steps: int,
        sampler_interpolation_coeff: float = 0.0,
        time_discretization: str = "uniform",
        time_discretization_kwargs: Dict[str, Any] = {},
        policy_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(
            env,
            diffuser,
            sampling_steps,
            sampler_interpolation_coeff,
            time_discretization,
            time_discretization_kwargs,
            policy_kwargs,
        )
        self.guidance_iterations = policy_kwargs["guidance_iterations"]
        self.guidance_lr = policy_kwargs["guidance_lr"]
        self.guidance_start_fraction = policy_kwargs["guidance_start_fraction"]
        self.projection_start_fraction = policy_kwargs["projection_start_fraction"]

    def formulate(self, conditions: Dict[Any, Any]):
        dofs = self.env.number_dofs(conditions)

        # Decision variable and parameters as MX symbolics
        x_sym = cs.MX.sym("x", dofs)
        n_cond_params = len([k for k in conditions if isinstance(k, int)]) * self.state_dim
        x_ref_sym = cs.MX.sym("x_ref", dofs)
        cond_sym = cs.MX.sym("cond", n_cond_params)

        # Pack all parameters into one vector: [x_ref; cond_params]
        p_sym = cs.vertcat(x_ref_sym, cond_sym)

        # Reconstruct conditions dict from flattened params
        conditions_params = {}
        offset = 0
        for k in sorted(k for k in conditions if isinstance(k, int)):
            conditions_params[k] = cond_sym[offset : offset + self.state_dim]
            offset += self.state_dim

        ## Constraints
        ineq_constraints, eq_constraints = self.env.generate_casadi_constraints(
            x_sym, conditions_params, normalize=False
        )

        g_list = []
        lbg_list = []
        ubg_list = []
        for constraint in ineq_constraints:
            dep = vec_dep(constraint, x_sym)
            if dep.shape[0] > 0:
                g_list.append(dep)
                lbg_list.extend([0.0] * int(dep.shape[0]))
                ubg_list.extend([1e20] * int(dep.shape[0]))
        for constraint in eq_constraints:
            dep = vec_dep(constraint, x_sym)
            if dep.shape[0] > 0:
                g_list.append(dep)
                lbg_list.extend([0.0] * int(dep.shape[0]))
                ubg_list.extend([0.0] * int(dep.shape[0]))

        g = cs.vertcat(*g_list) if g_list else cs.MX(0, 1)
        f = 0.5 * cs.sumsqr(x_sym - x_ref_sym)

        solver = cs.nlpsol(
            "projection",
            "ipopt",
            {"x": x_sym, "f": f, "g": g, "p": p_sym},
            self.ipopt_options,
        )

        self.nlp_solver = solver
        self.dofs = dofs
        self.nlp_lbg = np.array(lbg_list)
        self.nlp_ubg = np.array(ubg_list)
        self.n_cond_params = n_cond_params

    def _solve_projection_batch(self, dofs_ref_batch, conditions):
        """Solve projection NLP for each batch element sequentially."""
        batch_size = dofs_ref_batch.shape[0]

        # Pack condition parameters (same for all batch elements)
        cond_flat = np.concatenate(
            [to_np(conditions[k]).flatten() for k in sorted(k for k in conditions if isinstance(k, int))]
        )

        # Build parameter matrix [n_params, batch_size]
        p_batch = np.column_stack([
            np.concatenate([dofs_ref_batch[b], cond_flat])
            for b in range(batch_size)
        ])
        x0_batch = dofs_ref_batch.T  # [dofs, batch_size]

        lbg = self.nlp_lbg
        ubg = self.nlp_ubg

        xs = []
        for b in range(batch_size):
            sol = self.nlp_solver(x0=x0_batch[:, b], p=p_batch[:, b], lbg=lbg, ubg=ubg)
            self.record_ipopt_solve(self.nlp_solver.stats())
            xs.append(np.array(sol["x"]).flatten())

        return np.stack(xs)  # [batch_size, dofs]

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], batch_size: int = 1, **kwargs) -> Dict[str, Any]:
        del kwargs

        result_dict = {}
        timesteps = self.get_timesteps(self.sampling_steps)
        conditions = self.preprocess_conditions(conditions)

        ## Formulate the CaSaDi problem if necessary
        if not hasattr(self, "nlp_solver"):
            self.formulate(conditions)
        self.reset_ipopt_stats()

        # START
        intermediates = []
        data_estimates = []
        start_time = perf_counter()

        x = torch.randn(
            batch_size,
            self.diffuser.horizon,
            self.transition_dim,  # [Transition-dim]
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )
        total_control = 0.0
        total_projection_control = 0.0
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for i in range(self.sampling_steps):
            t = timesteps[i]
            s = timesteps[i + 1]

            # reference to compute control norm
            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)
            data_estimates.append(x1.detach().clone())

            epsilon = torch.randn_like(x)
            x_uncontrolled = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )

            apply_gradient_guidance = (i / self.sampling_steps) >= self.guidance_start_fraction
            if apply_gradient_guidance:
                for _ in range(self.guidance_iterations):
                    with torch.enable_grad():
                        inputs = x.detach().clone().requires_grad_(True)
                        model_out = self.diffuser.forward(inputs, t, conditions=conditions)
                        data_estimate = self.diffuser.estimate_data(model_out, inputs, t)
                        loss = -self.env.proxy_value(data_estimate).sum()
                        (g,) = torch.autograd.grad(loss, inputs, create_graph=False)
                    g = apply_conditioning(
                        g.detach(),
                        conditions=conditions,
                        action_dim=self.action_dim,
                        zero_conditioning=True,
                    )
                    x = x - self.guidance_lr * g
                    x = x.detach()

            model_out = self.diffuser.forward(x, t, conditions=conditions)
            xn_ref = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )

            ## Projection
            apply_projection = i / (self.sampling_steps - 1) >= self.projection_start_fraction

            if apply_projection:
                # Convert each batch element to DOFs
                dofs_ref_batch = np.stack([
                    to_np(self.env.matrix_to_dofs(xn_ref[b : b + 1], conditions).flatten())
                    for b in range(batch_size)
                ])  # [batch_size, dofs]

                xn_dofs_batch = self._solve_projection_batch(dofs_ref_batch, conditions)

                # Convert back to trajectory matrices
                xn_list = []
                for b in range(batch_size):
                    xn_dofs = torch.as_tensor(
                        xn_dofs_batch[b], dtype=xn_ref.dtype, device=xn_ref.device
                    )
                    xn_list.append(self.env.dofs_to_matrix(xn_dofs, conditions=conditions))
                xn = torch.stack(xn_list, dim=0)
            else:
                xn = xn_ref

            intermediates.append(xn.detach().clone())

            # Compute control
            # x - x_uncontrolled = u Delta T
            # total_norm = sum( ||u||*2 Delta T ) = sum(||x - x_uncontrolled||^2 / Delta T)
            control = (xn - x_uncontrolled).square().sum()
            total_control += control / (s - t)
            projection_control = (xn - xn_ref).square().sum()
            total_projection_control += projection_control / (s - t)
            x = xn

        data_estimates.append(x.detach().clone())

        # Compute proxy env value
        value = self.env.proxy_value(x)

        # END
        end_time = perf_counter()
        computation_time = end_time - start_time
        result_dict.update(
            {
                "computation_time": computation_time,
                "observation": x[:, :, self.action_dim :].clone(),
                "actions": x[:, :, : self.action_dim].clone(),
                "intermediates": torch.stack(intermediates, dim=1),
                "data_estimates": torch.stack(data_estimates, dim=1),
                "proxy_value": value,
                "total_control": total_control,
                "total_projection_control": total_projection_control,
            }
        )
        self.post_process_trajectory_data(trajectory_data=result_dict)
        return result_dict


# Adapted from Constrained Diffusers: https://github.com/z7076/Constrained_Diffuser
# [Note]: parameters have been rescaled to match the new time convention
class _ConstrainedDiffuserBase(Policy):
    """Base for primal-dual and augmented Lagrangian methods.

    Enforces constraints via Lagrangian relaxation during reverse diffusion.
    Updates dual variables (Lagrange multipliers) at each step to penalize
    infeasibility, and modifies the trajectory via the Lagrangian gradient.

    Args (policy_kwargs):
        guidance_start_fraction: Fraction of steps before guidance begins.
        use_cbfs: Whether to apply CBFs on the trajectory time dimension.
        cbf_alpha: CBF decay rate for temporal constraints.
        relaxation_steps_at_source: Extra unguided steps at the start.
        relaxation_steps_at_target: Extra unguided steps at the end.
        learning_rate: Base learning rate for dual variable updates.
        initial_penalty: Starting penalty weight (for augmented Lagrangian).
        final_penalty: Final penalty weight (for augmented Lagrangian).
    """
    def __init__(
        self,
        env: Maze2dConstrainedEnv,
        diffuser: Diffusion,
        sampling_steps: int,
        sampler_interpolation_coeff: float = 0.0,
        time_discretization: str = "uniform",
        time_discretization_kwargs: Dict[str, Any] = {},
        policy_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(
            env,
            diffuser,
            sampling_steps,
            sampler_interpolation_coeff,
            time_discretization,
            time_discretization_kwargs,
            policy_kwargs,
        )
        self.guidance_start_fraction = policy_kwargs["guidance_start_fraction"]
        self.use_cbfs = policy_kwargs["use_cbfs"]
        self.cbf_alpha = policy_kwargs["cbf_alpha"]
        self.relaxation_steps_at_source = policy_kwargs["relaxation_steps_at_source"]
        self.relaxation_steps_at_target = policy_kwargs["relaxation_steps_at_target"]
        self.learning_rate = policy_kwargs["learning_rate"]
        self.initial_penalty = policy_kwargs["initial_penalty"]
        self.final_penalty = policy_kwargs["final_penalty"]
        total_steps = (
            self.sampling_steps + self.relaxation_steps_at_target + self.relaxation_steps_at_source
        )
        self.penalty_multiplier = (self.final_penalty / self.initial_penalty) ** (1.0 / total_steps)
        self.step_learning_rate = self.learning_rate / total_steps

        # Build the constraints function E[g] >= 0
        # Also build equality constraint function for dynamics (residual = 0)
        if self.use_cbfs:

            def constraint_function(x: torch.Tensor) -> torch.Tensor:
                # Note CBFs are applied (unlike SafeDiffuser) on the trajectory time dimesion
                ineq_constraints, _ = self.env.get_constraints(x, normalize=True)
                assert (
                    ineq_constraints
                ), f"Must have at least one inequality constraint for projection methods."
                cbfs = []
                for constraint in ineq_constraints:
                    constraint = constraint.reshape(constraint.shape[0], self.env.horizon, -1)
                    cbf = constraint[:, 1:] - (1 - self.cbf_alpha) * constraint[:, :-1]
                    cbfs.append(cbf.reshape(constraint.shape[0], -1))
                cbfs = torch.cat(cbfs, dim=-1)
                return cbfs

            self.constraint_function = constraint_function

        else:

            def constraint_function(x: torch.Tensor) -> torch.Tensor:
                ineq_constraints, _ = self.env.get_constraints(x, normalize=True)
                assert (
                    ineq_constraints
                ), f"Must have at least one inequality constraint for projection methods."
                ineq_constraints = torch.cat(ineq_constraints, dim=-1)
                return ineq_constraints

            self.constraint_function = constraint_function

        # Equality constraint function for dynamics: residual should be 0
        def eq_constraint_function(x: torch.Tensor) -> torch.Tensor:
            _, eq_constraints = self.env.get_constraints(x, normalize=True)
            if not eq_constraints:
                return torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)
            return torch.cat(eq_constraints, dim=-1)

        self.eq_constraint_function = eq_constraint_function

        assert self.cbf_alpha >= 0.0

    def primal_dual_samples(
        self, conditions: Dict[Any, Any], batch_size: int = 1
    ) -> Dict[str, Any]:
        result_dict = {}
        timesteps = self.get_timesteps(self.sampling_steps)
        conditions = self.preprocess_conditions(conditions)

        # START
        intermediates = []
        data_estimates = []
        start_time = perf_counter()

        x = torch.randn(
            batch_size,
            self.diffuser.horizon,
            self.transition_dim,  # [Transition-dim]
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )

        ## Initialize dual variables (inequality: clamped >= 0)
        constraint_eval = self.constraint_function(x)  # [B, H*C_ineq]
        dual_vars = torch.zeros_like(constraint_eval)

        ## Initialize equality dual variables (unconstrained) for dynamics
        eq_constraint_eval = self.eq_constraint_function(x)  # [B, H*C_eq]
        eq_dual_vars = torch.zeros_like(eq_constraint_eval)

        total_control = 0.0
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for iter in range(
            -self.relaxation_steps_at_source,
            self.sampling_steps + self.relaxation_steps_at_target,
        ):
            relaxation_iter = (iter < 0) or (iter >= self.sampling_steps)
            i = max(min(iter, self.sampling_steps - 1), 0)
            t = timesteps[i]
            s = timesteps[i + 1]

            # reference to compute control norm
            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)
            data_estimates.append(x1.detach().clone())

            epsilon = torch.randn_like(x)
            x_uncontrolled = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )

            apply_guidance = (i / self.sampling_steps) >= self.guidance_start_fraction
            if apply_guidance:
                ## Compute score function correction for Primal-Dual updates
                # 1. VJP between duals and constraint gradients (inequality)
                with torch.enable_grad():
                    x = x.clone().detach().requires_grad_(True)
                    lam = dual_vars.clone().detach()
                    constraints = self.constraint_function(x)
                    # Equality constraints: dynamics residual (should be = 0)
                    eq_constraints = self.eq_constraint_function(x)
                    eq_lam = eq_dual_vars.clone().detach()
                    # Combined gradient: ineq duals push g>=0, eq duals push h=0
                    total_lagrangian = (constraints * lam).sum()
                    if eq_constraints.shape[-1] > 0:
                        total_lagrangian = total_lagrangian + (eq_constraints * eq_lam).sum()
                    grad = torch.autograd.grad(total_lagrangian, x, create_graph=False)[0]
                    x = x.detach()

                ## Correct score and sample (Primal)
                xn = x_uncontrolled + grad

                ## Update Duals (inequality: clamped >= 0)
                dual_vars = (dual_vars - self.step_learning_rate * constraints.detach()).clamp(
                    min=0.0
                )
                ## Update Duals (equality: unconstrained, push residual to 0)
                if eq_constraints.shape[-1] > 0:
                    eq_dual_vars = eq_dual_vars + self.step_learning_rate * eq_constraints.detach()

            else:
                xn = x_uncontrolled

            intermediates.append(xn.detach().clone())

            # Compute control
            # x - x_uncontrolled = u Delta T
            # total_norm = sum( ||u||*2 Delta T ) = sum(||x - x_uncontrolled||^2 / Delta T)
            if not relaxation_iter:
                control = (xn - x_uncontrolled).square().sum()
                total_control += control / (s - t)
            x = xn

        data_estimates.append(x.detach().clone())

        # END
        end_time = perf_counter()
        computation_time = end_time - start_time

        # Compute proxy env value
        value = self.env.proxy_value(x)
        result_dict.update(
            {
                "computation_time": computation_time,
                "observation": x[:, :, self.action_dim :].clone(),
                "actions": x[:, :, : self.action_dim].clone(),
                "intermediates": torch.stack(intermediates, dim=1),
                "data_estimates": torch.stack(data_estimates, dim=1),
                "proxy_value": value,
                "total_control": total_control,
            }
        )
        self.post_process_trajectory_data(trajectory_data=result_dict)
        return result_dict

    def augmented_lagrangian_samples(
        self, conditions: Dict[Any, Any], batch_size: int = 1
    ) -> Dict[str, Any]:
        result_dict = {}
        timesteps = self.get_timesteps(self.sampling_steps)
        conditions = self.preprocess_conditions(conditions)

        # START
        intermediates = []
        data_estimates = []
        start_time = perf_counter()

        x = torch.randn(
            batch_size,
            self.diffuser.horizon,
            self.transition_dim,  # [Transition-dim]
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )

        ## Initialize dual and slack variables (inequality)
        constraint_eval = self.constraint_function(x)  # [B, H*C_ineq]
        dual_vars = torch.zeros_like(constraint_eval)
        slack_vars = torch.zeros_like(dual_vars)
        penalty = self.initial_penalty

        ## Initialize equality dual variables (unconstrained) for dynamics
        eq_constraint_eval = self.eq_constraint_function(x)  # [B, H*C_eq]
        eq_dual_vars = torch.zeros_like(eq_constraint_eval)

        total_control = 0.0
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for iter in range(
            -self.relaxation_steps_at_source,
            self.sampling_steps + self.relaxation_steps_at_target,
        ):
            relaxation_iter = (iter < 0) or (iter >= self.sampling_steps)
            i = max(min(iter, self.sampling_steps - 1), 0)
            t = timesteps[i]
            s = timesteps[i + 1]

            # reference to compute control norm
            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)
            data_estimates.append(x1.detach().clone())

            epsilon = torch.randn_like(x)
            x_uncontrolled = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )

            apply_guidance = (i / self.sampling_steps) >= self.guidance_start_fraction
            if apply_guidance:
                ## Compute score function correction for Augmented Lagrangian updates
                # 1. VJP between duals and constraint gradients
                with torch.enable_grad():
                    x = x.clone().detach().requires_grad_(True)
                    lam = dual_vars.clone().detach()
                    constraints = self.constraint_function(x)
                    constrain_det = constraints.clone().detach()
                    vec = lam + penalty * (constrain_det - slack_vars)
                    # Inequality Lagrangian
                    total_lagrangian = (constraints * vec).sum()
                    # Equality constraints (dynamics): penalty + dual term
                    eq_constraints = self.eq_constraint_function(x)
                    if eq_constraints.shape[-1] > 0:
                        eq_lam = eq_dual_vars.clone().detach()
                        eq_det = eq_constraints.clone().detach()
                        eq_vec = eq_lam + penalty * eq_det
                        total_lagrangian = total_lagrangian + (eq_constraints * eq_vec).sum()
                    grad = torch.autograd.grad(total_lagrangian, x, create_graph=False)[0]
                    x = x.detach()

                ## Clamp gradient norm to prevent divergence
                grad_norm = grad.norm()
                max_grad_norm = x_uncontrolled.norm()
                if grad_norm > max_grad_norm:
                    grad = grad * (max_grad_norm / grad_norm)

                ## Correct score and sample (Primal)
                xn = x_uncontrolled - grad

                ## Update Duals, Slacks, Penalty (inequality)
                slack_vars = (dual_vars / penalty + constraints).clamp(min=0.0)
                dual_vars = dual_vars + penalty * (constraints - slack_vars)
                ## Update Duals (equality: unconstrained)
                if eq_constraints.shape[-1] > 0:
                    eq_dual_vars = eq_dual_vars + penalty * eq_constraints.detach()
                penalty *= self.penalty_multiplier
            else:
                xn = x_uncontrolled

            intermediates.append(xn.detach().clone())

            # Compute control
            # x - x_uncontrolled = u Delta T
            # total_norm = sum( ||u||*2 Delta T ) = sum(||x - x_uncontrolled||^2 / Delta T)
            if not relaxation_iter:
                control = (xn - x_uncontrolled).square().sum()
                total_control += control / (s - t)
            x = xn

        data_estimates.append(x.detach().clone())

        # END
        end_time = perf_counter()
        computation_time = end_time - start_time

        # Compute proxy env value
        value = self.env.proxy_value(x)
        result_dict.update(
            {
                "computation_time": computation_time,
                "observation": x[:, :, self.action_dim :].clone(),
                "actions": x[:, :, : self.action_dim].clone(),
                "intermediates": torch.stack(intermediates, dim=1),
                "data_estimates": torch.stack(data_estimates, dim=1),
                "proxy_value": value,
                "total_control": total_control,
            }
        )
        self.post_process_trajectory_data(trajectory_data=result_dict)
        return result_dict


# Adapted from Constrained Diffusers: https://github.com/z7076/Constrained_Diffuser
# [Note]: parameters have been rescaled to match the new time convention
class PrimalDualPolicy(_ConstrainedDiffuserBase):
    """Primal-dual constraint enforcement with direct dual variable updates."""

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], batch_size: int = 1, **kwargs) -> Dict[str, Any]:
        return self.primal_dual_samples(
            conditions=conditions, batch_size=batch_size, **kwargs
        )


# Adapted from Constrained Diffusers: https://github.com/z7076/Constrained_Diffuser
# [Note]: parameters have been rescaled to match the new time convention
class AugmentedLagrangianPolicy(_ConstrainedDiffuserBase):
    """Augmented Lagrangian with slack variables and geometric penalty scheduling."""

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], batch_size: int = 1, **kwargs) -> Dict[str, Any]:
        return self.augmented_lagrangian_samples(
            conditions=conditions, batch_size=batch_size, **kwargs
        )


# Adapted from SafeDiffuser: https://github.com/Weixy21/SafeDiffuser
# [Note]: parameters have been rescaled to match the new time convention
class _SafeDiffuserBase(Policy):
    """Base for CBF-constrained sampling (SafeDiffuser).

    Solves for the closest feasible velocity to the unconstrained denoising
    velocity at each step, subject to CBF constraints: J^T * v + k * h(x) >= 0.
    Uses QP (qpth) when only inequality constraints, IPOPT otherwise.
    Subclasses (RoS, ReS, TVS) override formulate_ipopt/formulate_qp.

    Args (policy_kwargs):
        cbf_coefficient: CBF gain k (controls constraint enforcement strength).
        guidance_start_fraction: Fraction of steps before guidance begins.
        allow_qp: Use QP solver when possible (faster than IPOPT).
        ipopt_options: Optional dict of IPOPT solver options merged over defaults.
    """
    def __init__(
        self,
        env: Maze2dConstrainedEnv,
        diffuser: Diffusion,
        sampling_steps: int,
        sampler_interpolation_coeff: float = 0.0,
        time_discretization: str = "uniform",
        time_discretization_kwargs: Dict[str, Any] = {},
        policy_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(
            env,
            diffuser,
            sampling_steps,
            sampler_interpolation_coeff,
            time_discretization,
            time_discretization_kwargs,
            policy_kwargs,
        )

        self.allow_qp = policy_kwargs["allow_qp"]
        self.cbf_coefficient = policy_kwargs["cbf_coefficient"]
        self.guidance_start_fraction = policy_kwargs["guidance_start_fraction"]

        ## Build the QP / IPOPT formulation
        # 1. If the constraints are only inequalities then CBFs constraints are linear,
        #    and the optimization problem reduces to a QP
        # 2. If there are equality constraints the problem may be non-linear and we solve it
        #    with Casadi IPOPT.
        x = torch.zeros(
            1,
            self.diffuser.horizon,
            self.transition_dim,  # [Transition-dim]
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )

        ineq_constraints, eq_constraints = self.env.get_constraints(x)
        self.num_ineq = sum([constraint.shape[-1] for constraint in ineq_constraints])
        self.num_eq = sum([constraint.shape[-1] for constraint in eq_constraints])
        assert self.num_ineq, f"SafeDiffuser requires at least one inequality constraint"
        self.is_qp = self.allow_qp and self.num_eq == 0

    def formulate_ipopt(self, conditions: Dict[Any, Any]):
        raise NotImplementedError

    def formulate_qp(self, x_current: torch.Tensor, vel_ref: torch.Tensor, **kwargs):
        raise NotImplementedError

    def _build_dynamics_eq_constraints(self, x_current: torch.Tensor, dt_diff: float):
        """Build per-timestep dynamics equality constraints for QP: A_eq @ vel = b_eq.

        At each trajectory timestep t, constrains the state velocity so that
        x_next[t,obs] moves toward the dynamics-predicted state:
            vel[t, obs] = (A @ s_t + B @ a_t + c - s_t) / dt_diff
        """
        B, H, D = x_current.shape
        if self.env.dynamics_model is None:
            return None, None

        A_dyn = torch.tensor(self.env.dynamics_model["A"], device=x_current.device, dtype=x_current.dtype)
        B_dyn = torch.tensor(self.env.dynamics_model["B"], device=x_current.device, dtype=x_current.dtype)
        c_dyn = torch.tensor(self.env.dynamics_model["c"], device=x_current.device, dtype=x_current.dtype)

        s = x_current[:, :, self.env.action_dim:]   # (B, H, state_dim)
        a = x_current[:, :, :self.env.action_dim]    # (B, H, action_dim)
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim

        # Dynamics-predicted velocity for state components: (A@s + B@a + c - s) / dt
        pred_next = s @ A_dyn.T + a @ B_dyn.T + c_dyn.view(1, 1, -1)  # (B, H, state_dim)
        b_eq = ((pred_next - s) / dt_diff).reshape(B * H, state_dim)   # (B*H, state_dim)

        # A_eq selects state velocity components: [0_{sd,ad} | I_{sd}]
        A_eq = torch.zeros(B * H, state_dim, D, device=x_current.device, dtype=x_current.dtype)
        A_eq[:, :, action_dim:action_dim + state_dim] = torch.eye(
            state_dim, device=x_current.device, dtype=x_current.dtype
        )

        return A_eq, b_eq

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], batch_size: int = 1, **kwargs) -> Dict[str, Any]:
        del kwargs

        result_dict = {}
        timesteps = self.get_timesteps(self.sampling_steps)
        conditions = self.preprocess_conditions(conditions)

        if not self.is_qp and not hasattr(self, "cs_optim"):
            self.formulate_ipopt(conditions=conditions)
            # Set parameters
            for k, v in conditions.items():
                self.cs_optim.set_value(
                    self.conditions_params[k], to_np(v.flatten())  # type: ignore
                )
        self.reset_ipopt_stats()

        # START
        intermediates = []
        data_estimates = []
        start_time = perf_counter()

        x = torch.randn(
            batch_size,
            self.diffuser.horizon,
            self.transition_dim,  # [Transition-dim]
            device=self.device,
            dtype=self.dtype,  # type: ignore
        )
        total_control = 0.0
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for i in range(self.sampling_steps):
            t = timesteps[i]
            s = timesteps[i + 1]

            # reference to compute control norm
            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)
            data_estimates.append(x1.detach().clone())

            epsilon = torch.randn_like(x)
            x_uncontrolled = self.sampler.integrate_step(
                x,
                t,
                s,
                model_out,
                conditions,
                epsilon=epsilon,
                last_step=i == (self.sampling_steps - 1),
            )
            xn_ref = x_uncontrolled

            ## CBF projection
            apply_guidance = (i / self.sampling_steps) >= self.guidance_start_fraction

            if apply_guidance:
                if not self.is_qp:
                    # IPOPT path: solve per batch element sequentially
                    xn_list = []
                    for b in range(batch_size):
                        x_current_dofs = to_np(self.env.matrix_to_dofs(x[b:b+1], conditions).flatten())
                        xn_ref_dofs = to_np(self.env.matrix_to_dofs(xn_ref[b:b+1], conditions).flatten())

                        delta_t = to_np((s - t))
                        self.cs_optim.set_value(self.delta_t, delta_t)
                        self.cs_optim.set_value(self.x_current, x_current_dofs)
                        self.cs_optim.set_value(self.x_next_ref, xn_ref_dofs)
                        self.cs_optim.set_initial(self.x_next, xn_ref_dofs)

                        self._set_ipopt_params(i, t)

                        try:
                            solution = self.cs_optim.solve_limited()
                            xn_dofs = solution.value(self.x_next)
                            self.record_ipopt_solve(self.cs_optim.stats())
                        except RuntimeError as e:
                            print(f"Solver failed with error: {e}.\nReturning last available value.")
                            xn_dofs = self.cs_optim.debug.value(self.x_next)
                            try:
                                self.record_ipopt_solve(self.cs_optim.debug.stats(), failed=True)
                            except Exception:
                                self.record_ipopt_solve(None, failed=True)

                        xn_dofs = torch.as_tensor(xn_dofs, dtype=xn_ref.dtype, device=xn_ref.device)
                        xn_list.append(self.env.dofs_to_matrix(xn_dofs, conditions=conditions))
                    xn = torch.stack(xn_list, dim=0)
                else:
                    ## Set-up QP problem with qpth.qp
                    qp_kwargs = self._get_qp_kwargs(i, t)

                    # QP solver
                    vel_ref = (xn_ref - x) / (s - t)
                    Q, q, G, h, A_eq, b_eq = self.formulate_qp(
                        x_current=x,
                        vel_ref=vel_ref,
                        dt_diff=float(s - t),
                        **qp_kwargs,
                    )
                    try:
                        vel_opt = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(
                            Q, q, G, h, A_eq, b_eq
                        )
                    except Exception as qp_err:
                        print(f"PDIPM_BATCHED solver failed with {qp_err}")
                        vel_opt = vel_ref

                    # Filter out slack variables (if any)
                    vel_opt = vel_opt[..., : self.transition_dim]

                    vel_opt = apply_conditioning(
                        vel_opt.reshape(batch_size, -1, self.transition_dim),
                        conditions=conditions,
                        zero_conditioning=True,
                    )
                    xn = x + vel_opt * (s - t)
            else:
                xn = xn_ref

            intermediates.append(xn.detach().clone())

            # Compute control
            # x - x_uncontrolled = u Delta T
            # total_norm = sum( ||u||*2 Delta T ) = sum(||x - x_uncontrolled||^2 / Delta T)
            control = (xn - x_uncontrolled).square().sum()
            total_control += control / (s - t)
            x = xn

        data_estimates.append(x.detach().clone())

        # Compute proxy env value
        value = self.env.proxy_value(x)

        # END
        end_time = perf_counter()
        computation_time = end_time - start_time
        result_dict.update(
            {
                "computation_time": computation_time,
                "observation": x[:, :, self.action_dim :].clone(),
                "actions": x[:, :, : self.action_dim].clone(),
                "intermediates": torch.stack(intermediates, dim=1),
                "data_estimates": torch.stack(data_estimates, dim=1),
                "proxy_value": value,
                "total_control": total_control,
            }
        )
        self.post_process_trajectory_data(trajectory_data=result_dict)
        return result_dict

    def _set_ipopt_params(self, i: int, t: torch.Tensor):
        """Set algorithm-specific IPOPT parameters per step. Override in subclasses."""
        pass

    def _get_qp_kwargs(self, i: int, t: torch.Tensor) -> Dict[str, Any]:
        """Return algorithm-specific kwargs for formulate_qp. Override in subclasses."""
        return {}


# Adapted from SafeDiffuser: https://github.com/Weixy21/SafeDiffuser
# [Note]: parameters have been rescaled to match the new time convention
class RoSPolicy(_SafeDiffuserBase):
    """Reconstruction Sampling (RoS): standard CBF constraint J^T * v + k * h(x) >= 0."""

    algorithm_type = "RoS"

    def formulate_ipopt(self, conditions: Dict[Any, Any]):
        dofs = self.env.number_dofs(conditions)
        cs_optim = cs.Opti()
        conditions_optim = {k: cs_optim.parameter(self.state_dim) for k in conditions.keys()}
        x_current = cs_optim.parameter(dofs)
        x_next_ref = cs_optim.parameter(dofs)
        delta_t = cs_optim.parameter(1)
        x_next = cs_optim.variable(dofs)
        conditions_params = {k: cs.MX.sym(f"conditions_params_{k}", self.state_dim, 1) for k in conditions.keys()}  # type: ignore
        cs_dofs = cs.MX.sym("cs_dofs", dofs, 1)  # type: ignore
        ## Constraints ##
        ineq_constraints, eq_constraints = self.env.generate_casadi_constraints(cs_dofs, conditions_params=conditions_params, normalize=False)
        # Equality constraints -> As-is
        for i, constraint in enumerate(eq_constraints):
            inputs = [cs_dofs, *flatten_conditions(conditions_params)]
            constraint_fn = cs.Function(f"eq_constraint_{i}", inputs, [constraint])
            constraint_optim = constraint_fn(x_next, *flatten_conditions(conditions_optim))
            cs_optim.subject_to(constraint_optim == 0)
        # Inequality constraints -> CBFs
        velocity = (x_next - x_current) / delta_t
        velocity_ref = (x_next_ref - x_current) / delta_t
        for i, constraint in enumerate(ineq_constraints):
            inputs = [cs_dofs, *flatten_conditions(conditions_params)]
            constraint_fn = cs.Function(f"ineq_constraint_{i}", inputs, [constraint])
            jacobian = cs.jacobian(constraint, cs_dofs)
            jacobian_fn = cs.Function(f"jacobian_constraint_{i}", inputs, [jacobian])
            constraint_optim = constraint_fn(x_current, *flatten_conditions(conditions_optim)).reshape((-1, 1))  # type: ignore
            jacobian_optim = jacobian_fn(x_current, *flatten_conditions(conditions_optim))
            # RoS: Gradient h^T velocity + k * h >= 0
            cbf_constraint = (cs.mtimes(jacobian_optim, velocity) + self.cbf_coefficient * constraint_optim)
            cs_optim.subject_to(vec_dep(cbf_constraint, x_next) >= 0)
        ## Objective
        projection_cost = cs.sumsqr(velocity - velocity_ref)
        cs_optim.minimize(projection_cost)
        cs_optim.solver("ipopt", self.ipopt_options)
        self.cs_optim = cs_optim
        self.delta_t = delta_t
        self.x_current = x_current
        self.x_next_ref = x_next_ref
        self.x_next = x_next
        self.control_cost = projection_cost
        self.conditions_params = conditions_optim

    def formulate_qp(self, x_current: torch.Tensor, vel_ref: torch.Tensor, **kwargs):
        assert x_current.ndim == 3
        B, H, D = x_current.shape
        def f(x_in: torch.Tensor) -> torch.Tensor:
            ineq_list, _ = self.env.get_constraints(x_in, normalize=True)
            ineq = torch.cat([c.reshape(B, H, -1) for c in ineq_list], dim=-1)
            return ineq
        x_var = x_current.detach().clone().requires_grad_(True)
        ineq, vjp_fn = torch.func.vjp(f, x_var)  # type: ignore
        NC = ineq.shape[-1]
        E = torch.eye(NC, device=ineq.device, dtype=ineq.dtype)
        V = E[:, None, None, :].expand(NC, B, H, NC).contiguous()
        grads = torch.func.vmap(lambda v: vjp_fn(v)[0])(V)
        jacobians = grads.permute(1, 2, 0, 3).contiguous()
        jacobians = jacobians.reshape(B * H, NC, D)
        constraints = ineq.detach().reshape(B * H, NC)
        vel_ref = vel_ref.reshape(B * H, D)
        # -J^T x <= k * constraint
        Q = torch.autograd.Variable(torch.eye(D, device=x_current.device, dtype=x_current.dtype).expand(B * H, D, D))
        q = -vel_ref
        G = -jacobians
        h = self.cbf_coefficient * constraints
        e = torch.autograd.Variable(torch.Tensor())
        # Dynamics equality constraints if available
        dt_diff = kwargs.get("dt_diff", None)
        if dt_diff is not None:
            A_eq, b_eq = self._build_dynamics_eq_constraints(x_current, dt_diff)
        else:
            A_eq, b_eq = None, None
        if A_eq is not None:
            n_eq = A_eq.shape[1]
            return Q, q, G, h, A_eq, b_eq  # type: ignore
        return Q, q, G, h, e, e  # type: ignore


# Adapted from SafeDiffuser: https://github.com/Weixy21/SafeDiffuser
# [Note]: parameters have been rescaled to match the new time convention
class ReSPolicy(_SafeDiffuserBase):
    """Relaxed Sampling (ReS): CBF with slack variables J^T * v + k * h - w * r >= 0.

    Args (policy_kwargs):
        slack_weight: Penalty weight for slack variables in the objective.
        relaxation_end_fraction: Fraction of steps after which slack is disabled.
    """

    algorithm_type = "ReS"

    def __init__(self, env, diffuser, sampling_steps, sampler_interpolation_coeff=0.0,
                 time_discretization="uniform", time_discretization_kwargs={}, policy_kwargs={}):
        super().__init__(env, diffuser, sampling_steps, sampler_interpolation_coeff,
                         time_discretization, time_discretization_kwargs, policy_kwargs)
        self.slack_weight = policy_kwargs["slack_weight"]
        self.relaxation_end_fraction = policy_kwargs["relaxation_end_fraction"]

    def formulate_ipopt(self, conditions: Dict[Any, Any]):
        dofs = self.env.number_dofs(conditions)
        cs_optim = cs.Opti()
        conditions_optim = {k: cs_optim.parameter(self.state_dim) for k in conditions.keys()}
        x_current = cs_optim.parameter(dofs)
        x_next_ref = cs_optim.parameter(dofs)
        delta_t = cs_optim.parameter(1)
        x_next = cs_optim.variable(dofs)
        conditions_params = {k: cs.MX.sym(f"conditions_params_{k}", self.state_dim, 1) for k in conditions.keys()}  # type: ignore
        cs_dofs = cs.MX.sym("cs_dofs", dofs, 1)  # type: ignore
        ## Constraints ##
        ineq_constraints, eq_constraints = self.env.generate_casadi_constraints(cs_dofs, conditions_params=conditions_params, normalize=False)
        slacks = []
        relaxation_weight = cs_optim.parameter(1)
        self.relaxation_weight = relaxation_weight
        # Equality constraints -> As-is
        for i, constraint in enumerate(eq_constraints):
            inputs = [cs_dofs, *flatten_conditions(conditions_params)]
            constraint_fn = cs.Function(f"eq_constraint_{i}", inputs, [constraint])
            constraint_optim = constraint_fn(x_next, *flatten_conditions(conditions_optim))
            cs_optim.subject_to(constraint_optim == 0)
        # Inequality constraints -> CBFs
        velocity = (x_next - x_current) / delta_t
        velocity_ref = (x_next_ref - x_current) / delta_t
        for i, constraint in enumerate(ineq_constraints):
            inputs = [cs_dofs, *flatten_conditions(conditions_params)]
            constraint_fn = cs.Function(f"ineq_constraint_{i}", inputs, [constraint])
            jacobian = cs.jacobian(constraint, cs_dofs)
            jacobian_fn = cs.Function(f"jacobian_constraint_{i}", inputs, [jacobian])
            constraint_optim = constraint_fn(x_current, *flatten_conditions(conditions_optim)).reshape((-1, 1))  # type: ignore
            jacobian_optim = jacobian_fn(x_current, *flatten_conditions(conditions_optim))
            # ReS: Gradient h^T velocity + k * h - w * r >= 0
            slack_var = cs_optim.variable(constraint_optim.shape[0])  # type: ignore
            slacks.append(slack_var)
            cbf_constraint = (cs.mtimes(jacobian_optim, velocity) + self.cbf_coefficient * constraint_optim - relaxation_weight * slack_var)
            cs_optim.subject_to(vec_dep(cbf_constraint, x_next) >= 0)
        ## Objective
        projection_cost = cs.sumsqr(velocity - velocity_ref)
        projection_cost = projection_cost + cs.sumsqr(cs.vertcat(*slacks))
        self.slack_vars = slacks
        cs_optim.minimize(projection_cost)
        cs_optim.solver("ipopt", self.ipopt_options)
        self.cs_optim = cs_optim
        self.delta_t = delta_t
        self.x_current = x_current
        self.x_next_ref = x_next_ref
        self.x_next = x_next
        self.control_cost = projection_cost
        self.conditions_params = conditions_optim

    def _set_ipopt_params(self, i: int, t: torch.Tensor):
        slack_weight = float(self.slack_weight * ((i / self.sampling_steps) < self.relaxation_end_fraction))
        self.cs_optim.set_value(self.relaxation_weight, slack_weight)
        for slack in self.slack_vars:
            self.cs_optim.set_initial(slack, np.zeros(slack.shape))

    def _get_qp_kwargs(self, i: int, t: torch.Tensor) -> Dict[str, Any]:
        slack_weight = float(self.slack_weight * ((i / self.sampling_steps) < self.relaxation_end_fraction))
        return {"relaxation_weight": slack_weight}

    def formulate_qp(self, x_current: torch.Tensor, vel_ref: torch.Tensor, **kwargs):
        assert x_current.ndim == 3
        B, H, D = x_current.shape
        def f(x_in: torch.Tensor) -> torch.Tensor:
            ineq_list, _ = self.env.get_constraints(x_in, normalize=True)
            ineq = torch.cat([c.reshape(B, H, -1) for c in ineq_list], dim=-1)
            return ineq
        x_var = x_current.detach().clone().requires_grad_(True)
        ineq, vjp_fn = torch.func.vjp(f, x_var)  # type: ignore
        NC = ineq.shape[-1]
        E = torch.eye(NC, device=ineq.device, dtype=ineq.dtype)
        V = E[:, None, None, :].expand(NC, B, H, NC).contiguous()
        grads = torch.func.vmap(lambda v: vjp_fn(v)[0])(V)
        jacobians = grads.permute(1, 2, 0, 3).contiguous()
        jacobians = jacobians.reshape(B * H, NC, D)
        constraints = ineq.detach().reshape(B * H, NC)
        vel_ref = vel_ref.reshape(B * H, D)
        # -J^T x + w * r <= k * constraint
        assert "relaxation_weight" in kwargs
        Q = torch.autograd.Variable(torch.eye(D + NC, device=x_current.device, dtype=x_current.dtype).expand(B * H, D + NC, D + NC))
        G = torch.cat([-jacobians, kwargs["relaxation_weight"] * torch.eye(NC, NC, device=x_current.device, dtype=x_current.dtype).expand(B * H, NC, NC)], dim=-1)
        q = torch.cat([-vel_ref, torch.zeros(B * H, NC, device=vel_ref.device, dtype=vel_ref.dtype)], dim=-1)
        h = self.cbf_coefficient * constraints
        e = torch.autograd.Variable(torch.Tensor())
        # Dynamics equality constraints if available
        dt_diff = kwargs.get("dt_diff", None)
        if dt_diff is not None:
            A_eq, b_eq = self._build_dynamics_eq_constraints(x_current, dt_diff)
        else:
            A_eq, b_eq = None, None
        if A_eq is not None:
            n_eq = A_eq.shape[1]
            # Extend A_eq for slack variables if present
            if Q.shape[-1] > D:
                NC_slack = Q.shape[-1] - D
                A_eq = torch.cat([A_eq, torch.zeros(B * H, n_eq, NC_slack, device=A_eq.device, dtype=A_eq.dtype)], dim=-1)
            return Q, q, G, h, A_eq, b_eq  # type: ignore
        return Q, q, G, h, e, e  # type: ignore


# Adapted from SafeDiffuser: https://github.com/Weixy21/SafeDiffuser
# [Note]: parameters have been rescaled to match the new time convention
class TVSPolicy(_SafeDiffuserBase):
    """Time-Varying Safety (TVS): CBF with sigmoid-based time-varying barrier margin.

    The barrier margin varies over the diffusion trajectory via a sigmoid function,
    allowing tighter constraint enforcement near the end of sampling.

    Args (policy_kwargs):
        sigmoid_bias_fraction: Sigmoid center point as fraction of sampling steps.
        sigmoid_input_scale: Controls sigmoid steepness.
        sigmoid_output_scale: Scales the sigmoid output range.
    """

    algorithm_type = "TVS"

    def __init__(self, env, diffuser, sampling_steps, sampler_interpolation_coeff=0.0,
                 time_discretization="uniform", time_discretization_kwargs={}, policy_kwargs={}):
        super().__init__(env, diffuser, sampling_steps, sampler_interpolation_coeff,
                         time_discretization, time_discretization_kwargs, policy_kwargs)
        self.sigmoid_time_bias = policy_kwargs["sigmoid_bias_fraction"]
        self.sigmoid_input_scale = policy_kwargs["sigmoid_input_scale"]
        self.sigmoid_output_scale = policy_kwargs["sigmoid_output_scale"]

    def formulate_ipopt(self, conditions: Dict[Any, Any]):
        dofs = self.env.number_dofs(conditions)
        cs_optim = cs.Opti()
        conditions_optim = {k: cs_optim.parameter(self.state_dim) for k in conditions.keys()}
        x_current = cs_optim.parameter(dofs)
        x_next_ref = cs_optim.parameter(dofs)
        delta_t = cs_optim.parameter(1)
        x_next = cs_optim.variable(dofs)
        conditions_params = {k: cs.MX.sym(f"conditions_params_{k}", self.state_dim, 1) for k in conditions.keys()}  # type: ignore
        cs_dofs = cs.MX.sym("cs_dofs", dofs, 1)  # type: ignore
        ## Constraints ##
        ineq_constraints, eq_constraints = self.env.generate_casadi_constraints(cs_dofs, conditions_params=conditions_params, normalize=False)
        barrier_margin = cs_optim.parameter(1)
        barrier_margin_diff = cs_optim.parameter(1)
        self.barrier_margin = barrier_margin
        self.barrier_margin_diff = barrier_margin_diff
        # Equality constraints -> As-is
        for i, constraint in enumerate(eq_constraints):
            inputs = [cs_dofs, *flatten_conditions(conditions_params)]
            constraint_fn = cs.Function(f"eq_constraint_{i}", inputs, [constraint])
            constraint_optim = constraint_fn(x_next, *flatten_conditions(conditions_optim))
            cs_optim.subject_to(constraint_optim == 0)
        # Inequality constraints -> CBFs
        velocity = (x_next - x_current) / delta_t
        velocity_ref = (x_next_ref - x_current) / delta_t
        for i, constraint in enumerate(ineq_constraints):
            inputs = [cs_dofs, *flatten_conditions(conditions_params)]
            constraint_fn = cs.Function(f"ineq_constraint_{i}", inputs, [constraint])
            jacobian = cs.jacobian(constraint, cs_dofs)
            jacobian_fn = cs.Function(f"jacobian_constraint_{i}", inputs, [jacobian])
            constraint_optim = constraint_fn(x_current, *flatten_conditions(conditions_optim)).reshape((-1, 1))  # type: ignore
            jacobian_optim = jacobian_fn(x_current, *flatten_conditions(conditions_optim))
            # TVS: Gradient h^T velocity + k * (h - sigmoid) + dsigmoid >= 0 (after time inversion)
            cbf_constraint = (cs.mtimes(jacobian_optim, velocity) + self.cbf_coefficient * (constraint_optim - barrier_margin) + barrier_margin_diff)  # type: ignore
            cs_optim.subject_to(vec_dep(cbf_constraint, x_next) >= 0)
        ## Objective
        projection_cost = cs.sumsqr(velocity - velocity_ref)
        cs_optim.minimize(projection_cost)
        cs_optim.solver("ipopt", self.ipopt_options)
        self.cs_optim = cs_optim
        self.delta_t = delta_t
        self.x_current = x_current
        self.x_next_ref = x_next_ref
        self.x_next = x_next
        self.control_cost = projection_cost
        self.conditions_params = conditions_optim

    def _set_ipopt_params(self, i: int, t: torch.Tensor):
        sigmoid = self.sigmoid_output_scale * torch.sigmoid(self.sigmoid_input_scale * (t - self.sigmoid_time_bias))
        sigmoid_diff = (self.sigmoid_input_scale * sigmoid * (1.0 - sigmoid / self.sigmoid_output_scale))
        barrier_margin = to_np((sigmoid - 1.0))
        barrier_margin_diff = to_np(sigmoid_diff)
        self.cs_optim.set_value(self.barrier_margin, barrier_margin)
        self.cs_optim.set_value(self.barrier_margin_diff, barrier_margin_diff)

    def _get_qp_kwargs(self, i: int, t: torch.Tensor) -> Dict[str, Any]:
        sigmoid = self.sigmoid_output_scale * torch.sigmoid(self.sigmoid_input_scale * (t - self.sigmoid_time_bias))
        sigmoid_diff = (self.sigmoid_input_scale * sigmoid * (1.0 - sigmoid / self.sigmoid_output_scale))
        barrier_margin = sigmoid - 1.0
        barrier_margin_diff = sigmoid_diff
        return {"barrier_margin": barrier_margin, "barrier_margin_diff": barrier_margin_diff}

    def formulate_qp(self, x_current: torch.Tensor, vel_ref: torch.Tensor, **kwargs):
        assert x_current.ndim == 3
        B, H, D = x_current.shape
        def f(x_in: torch.Tensor) -> torch.Tensor:
            ineq_list, _ = self.env.get_constraints(x_in, normalize=True)
            ineq = torch.cat([c.reshape(B, H, -1) for c in ineq_list], dim=-1)
            return ineq
        x_var = x_current.detach().clone().requires_grad_(True)
        ineq, vjp_fn = torch.func.vjp(f, x_var)  # type: ignore
        NC = ineq.shape[-1]
        E = torch.eye(NC, device=ineq.device, dtype=ineq.dtype)
        V = E[:, None, None, :].expand(NC, B, H, NC).contiguous()
        grads = torch.func.vmap(lambda v: vjp_fn(v)[0])(V)
        jacobians = grads.permute(1, 2, 0, 3).contiguous()
        jacobians = jacobians.reshape(B * H, NC, D)
        constraints = ineq.detach().reshape(B * H, NC)
        vel_ref = vel_ref.reshape(B * H, D)
        assert "barrier_margin" in kwargs
        assert "barrier_margin_diff" in kwargs
        # -J^T x <= k * constraint
        Q = torch.autograd.Variable(torch.eye(D, device=x_current.device, dtype=x_current.dtype).expand(B * H, D, D))
        q = -vel_ref
        G = -jacobians
        h = (self.cbf_coefficient * (constraints - kwargs["barrier_margin"]) + kwargs["barrier_margin_diff"])
        e = torch.autograd.Variable(torch.Tensor())
        # Dynamics equality constraints if available
        dt_diff = kwargs.get("dt_diff", None)
        if dt_diff is not None:
            A_eq, b_eq = self._build_dynamics_eq_constraints(x_current, dt_diff)
        else:
            A_eq, b_eq = None, None
        if A_eq is not None:
            n_eq = A_eq.shape[1]
            return Q, q, G, h, A_eq, b_eq  # type: ignore
        return Q, q, G, h, e, e  # type: ignore


class DiRecTPolicy(Policy):
    """Diffusion-based Receding-horizon ConTrol (DiRecT).

    Optimizes the terminal state prediction at each denoising step via IPOPT,
    balancing a regularization term (stay close to the diffusion prediction) with
    the environment objective (e.g. minimize path length).

    Args (policy_kwargs):
        guidance_start_fraction: Fraction of steps before optimization begins.
        control_penalty_weight: Regularization weight lambda for the MPC objective.
        ipopt_options: Optional dict of IPOPT solver options merged over defaults.
    """

    def __init__(self, env, diffuser, sampling_steps, sampler_interpolation_coeff=0.0,
                 time_discretization="uniform", time_discretization_kwargs={}, policy_kwargs={}):
        super().__init__(env, diffuser, sampling_steps, sampler_interpolation_coeff,
                         time_discretization, time_discretization_kwargs, policy_kwargs)
        self.guidance_start_fraction = policy_kwargs["guidance_start_fraction"]
        self.control_penalty_weight = policy_kwargs["control_penalty_weight"]

    def formulate(self, conditions: Dict[Any, Any]):
        dofs = self.env.number_dofs(conditions)
        x_sym = cs.MX.sym("x", dofs)
        x_ref_sym = cs.MX.sym("x_ref", dofs)
        reg_weight_sym = cs.MX.sym("reg_weight", 1)
        n_cond_params = len([k for k in conditions if isinstance(k, int)]) * self.state_dim
        cond_sym = cs.MX.sym("cond", n_cond_params)
        p_sym = cs.vertcat(x_ref_sym, reg_weight_sym, cond_sym)
        conditions_params = {}
        offset = 0
        for k in sorted(k for k in conditions if isinstance(k, int)):
            conditions_params[k] = cond_sym[offset : offset + self.state_dim]
            offset += self.state_dim
        ## Constraints
        ineq_constraints, eq_constraints = self.env.generate_casadi_constraints(x_sym, conditions_params, normalize=False)
        g_list = []
        lbg_list = []
        ubg_list = []
        for constraint in ineq_constraints:
            dep = vec_dep(constraint, x_sym)
            if dep.shape[0] > 0:
                g_list.append(dep)
                lbg_list.extend([0.0] * int(dep.shape[0]))
                ubg_list.extend([1e20] * int(dep.shape[0]))
        for constraint in eq_constraints:
            dep = vec_dep(constraint, x_sym)
            if dep.shape[0] > 0:
                g_list.append(dep)
                lbg_list.extend([0.0] * int(dep.shape[0]))
                ubg_list.extend([0.0] * int(dep.shape[0]))
        g = cs.vertcat(*g_list) if g_list else cs.MX(0, 1)
        ## Objective
        env_objective = self.env.generate_casadi_objective(x_sym, conditions_params)
        f = 0.5 * reg_weight_sym * cs.sumsqr(x_sym - x_ref_sym) - env_objective
        solver = cs.nlpsol("prediction_mpc", "ipopt", {"x": x_sym, "f": f, "g": g, "p": p_sym}, self.ipopt_options)
        self.nlp_solver = solver
        self.dofs = dofs
        self.nlp_lbg = np.array(lbg_list)
        self.nlp_ubg = np.array(ubg_list)
        self.n_cond_params = n_cond_params

    def _solve_mpc_batch(self, dofs_ref_batch, reg_weights, conditions):
        """Solve MPC NLP for each batch element sequentially."""
        batch_size = dofs_ref_batch.shape[0]
        cond_flat = np.concatenate([to_np(conditions[k]).flatten() for k in sorted(k for k in conditions if isinstance(k, int))])
        p_batch = np.column_stack([np.concatenate([dofs_ref_batch[b], [reg_weights[b]], cond_flat]) for b in range(batch_size)])
        x0_batch = dofs_ref_batch.T
        lbg = self.nlp_lbg
        ubg = self.nlp_ubg
        xs = []
        for b in range(batch_size):
            sol = self.nlp_solver(x0=x0_batch[:, b], p=p_batch[:, b], lbg=lbg, ubg=ubg)
            self.record_ipopt_solve(self.nlp_solver.stats())
            xs.append(np.array(sol["x"]).flatten())
        return np.stack(xs)

    @torch.no_grad()
    def generate(self, conditions: Dict[Any, Any], batch_size: int = 1, **kwargs) -> Dict[str, Any]:
        del kwargs
        result_dict = {}
        timesteps = self.get_timesteps(self.sampling_steps)
        conditions = self.preprocess_conditions(conditions)
        ## Formulate the CaSaDi problem if necessary
        if not hasattr(self, "nlp_solver"):
            self.formulate(conditions)
        self.reset_ipopt_stats()
        # START
        intermediates = []
        data_estimates = []
        start_time = perf_counter()
        x = torch.randn(batch_size, self.diffuser.horizon, self.transition_dim, device=self.device, dtype=self.dtype)  # type: ignore
        total_control = 0.0
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        intermediates.append(x.detach().clone())
        for i in range(self.sampling_steps):
            t = timesteps[i]
            s = timesteps[i + 1]
            # reference to compute control norm
            model_out = self.diffuser.forward(x, t, conditions=conditions)
            x1 = self.diffuser.estimate_data(model_out, x, t)
            data_estimates.append(x1.detach().clone())
            epsilon = torch.randn_like(x)
            x_uncontrolled = self.sampler.integrate_step(x, t, s, model_out, conditions, epsilon=epsilon, last_step=i == (self.sampling_steps - 1))
            xn_ref = x_uncontrolled
            ## Estimate proxy of final state after one-step horizon
            if i < self.sampling_steps - 1:
                model_out = self.diffuser.forward(xn_ref, s, conditions=conditions)
                terminal_prediction_ref = self.diffuser.estimate_data(model_out, xn_ref, s)
            else:
                terminal_prediction_ref = xn_ref
            ## Terminal MPC optimization
            apply_control = i / (self.sampling_steps - 1) >= self.guidance_start_fraction
            if apply_control:
                # regularization weight lambda * [alpha(s) **2 / Delta t]
                alpha_s = self.diffuser.noise_schedule.alpha_t(s)
                reg_weight_scalar = float(to_np(self.control_penalty_weight * alpha_s**2 / (s - t))) ####
                ## Test alpha_s() ** 2 / Delta t g ** 2
                g_s = self.diffuser.noise_schedule.forward_diffusion(s)
                reg_weight_scalar = float(to_np(self.control_penalty_weight * alpha_s**2 / (s - t) / g_s**2)) ####
                # Convert each batch element to DOFs
                dofs_ref_batch = np.stack([to_np(self.env.matrix_to_dofs(terminal_prediction_ref[b : b + 1], conditions).flatten()) for b in range(batch_size)])
                reg_weights = [reg_weight_scalar] * batch_size
                x_terminal_dofs_batch = self._solve_mpc_batch(dofs_ref_batch, reg_weights, conditions)
                # Convert back to trajectory matrices
                terminal_list = []
                for b in range(batch_size):
                    x_terminal_dofs = torch.as_tensor(x_terminal_dofs_batch[b], dtype=terminal_prediction_ref.dtype, device=terminal_prediction_ref.device)
                    terminal_list.append(self.env.dofs_to_matrix(x_terminal_dofs, conditions=conditions))
                terminal_prediction = torch.stack(terminal_list, dim=0)
                # Corrector step
                # xn_ctrl = xn_ref + alpha_t * (x1* - x1_ref)
                xn = xn_ref + alpha_s * (terminal_prediction - terminal_prediction_ref)
            else:
                xn = xn_ref
            intermediates.append(xn.detach().clone())
            # Compute control
            # x - x_uncontrolled = u Delta T
            # total_norm = sum( ||u||*2 Delta T ) = sum(||x - x_uncontrolled||^2 / Delta T)
            control = (xn - x_uncontrolled).square().sum()
            total_control += control / (s - t)
            x = xn
        data_estimates.append(x.detach().clone())
        # Compute proxy env value
        value = self.env.proxy_value(x)
        # END
        end_time = perf_counter()
        computation_time = end_time - start_time
        result_dict.update({"computation_time": computation_time, "observation": x[:, :, self.action_dim :].clone(), "actions": x[:, :, : self.action_dim].clone(), "intermediates": torch.stack(intermediates, dim=1), "data_estimates": torch.stack(data_estimates, dim=1), "proxy_value": value, "total_control": total_control})
        self.post_process_trajectory_data(trajectory_data=result_dict)
        return result_dict


POLICY_DICT = {
    "no-guidance": NoGuidancePolicy,
    "gradient-guidance": GradientGuidancePolicy,
    "classifier-guidance": ClassifierGuidancePolicy,
    "projection": ProjectionPolicy,
    "primal-dual": PrimalDualPolicy,
    "augmented-lagrangian": AugmentedLagrangianPolicy,
    "safediffuser-ros": RoSPolicy,
    "safediffuser-res": ReSPolicy,
    "safediffuser-tvs": TVSPolicy,
    "direct": DiRecTPolicy,
}
