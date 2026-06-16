"""
Temporal Unet implementation adapted from: https://github.com/Weixy21/SafeDiffuser/blob/main/diffuser/models/temporal.py

"""
import math
from typing import Callable, Any, Sequence, Union, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange


def to_np(x: Any) -> NDArray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) d -> b h c d", h=self.heads), qkv
        )
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c d -> b (h c) d")
        return self.to_out(out)


def extract(
    a: torch.Tensor, t: torch.Tensor, x_shape: Union[Sequence[int], torch.Size]
) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(
    x: torch.Tensor, conditions: Dict[Any, Any], action_dim: int
) -> torch.Tensor:
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


class WeightedLoss(nn.Module):

    def __init__(self, weights: torch.Tensor, action_dim: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim
        self._loss: Union[Callable, nn.Module]

    def forward(
        self, pred: torch.Tensor, targ: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # pred, targ : (batch_size, planning_horizon, transition_dim)
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]).mean()  # type: ignore
        return weighted_loss, {"a0_loss": a0_loss}


class ValueLoss(nn.Module):
    def __init__(self, *args):
        self._loss: Union[Callable, nn.Module]
        super().__init__()

    def forward(
        self, pred: torch.Tensor, targ: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                to_np(pred).squeeze(), to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.nan

        info = {
            "mean_pred": pred.mean(),
            "mean_targ": targ.mean(),
            "min_pred": pred.min(),
            "min_targ": targ.min(),
            "max_pred": pred.max(),
            "max_targ": targ.max(),
            "corr": corr,
        }

        return loss, info


class WeightedL1(WeightedLoss):

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, targ, reduction="none")


class ValueL1(ValueLoss):

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, targ, reduction="none")


LOSS_TYPES = {
    "l1": WeightedL1,
    "l2": WeightedL2,
    "value_l1": ValueL1,
    "value_l2": ValueL2,
}


class ResidualTemporalBlock(nn.Module):

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        horizon: int,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 32,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        attention: bool = False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in, dim_out, embed_dim=time_dim, horizon=horizon
                        ),
                        ResidualTemporalBlock(
                            dim_out, dim_out, embed_dim=time_dim, horizon=horizon
                        ),
                        (
                            Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                            if attention
                            else nn.Identity()
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in, embed_dim=time_dim, horizon=horizon
                        ),
                        (
                            Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                            if attention
                            else nn.Identity()
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time = t

        # x: (batch_size, planning_horizon, transition_dim)

        b, _, _ = x.shape
        while time.dim() > 1:  # remove tailing dimensions
            time = time[..., 0]
        if time.dim() == 0:
            time = time.repeat(b)
        # time: (batch_size,)

        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x


class WrappedFlowUnet(TemporalUnet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_info(self, horizon, transition_dim):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.vector_dim = horizon * transition_dim + 1

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch_size, planning_horizon * transition_dim + 1)
        # output: (batch_size, planning_horizon * transition_dim)

        batch_size, v_dim = src.shape

        time = src[:, -1:].squeeze(-1)  # (B, )

        src_3 = src[:, :-1]  # [B, T * C]
        src_3 = src_3.view(batch_size, self.horizon, self.transition_dim)  # [B, T, C]

        output_3 = super().forward(src_3, time)  # [B, T, C]
        output = output_3.reshape(batch_size, -1)

        return output

