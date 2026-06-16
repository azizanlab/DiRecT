import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
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

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
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


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
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

    def forward(self, x, t):
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# ---------------------------------------------------------------------------
# TemporalUnet
# ---------------------------------------------------------------------------

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 4, 8),
        attention=False,
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

    def forward(self, x, time):
        # x: (batch_size, horizon, transition_dim)
        b, _, _ = x.shape
        while time.dim() > 1:
            time = time[..., 0]
        if time.dim() == 0:
            time = time.repeat(b)

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


# ---------------------------------------------------------------------------
# Cosine schedule diffusion
# ---------------------------------------------------------------------------

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


class CosineScheduleDiffusion:
    """Continuous-time diffusion with cosine noise schedule.

    Convention:  t=0 is clean data, t=1 is pure noise.
    Training:    sample t ~ U[0,1], predict x_0 directly.
    Sampling:    reverse from t=1 to t=0 via DDPM steps.
    """

    def __init__(self, model, action_dim, s=0.008):
        self.model = model
        self.action_dim = action_dim
        self.s = s
        self._alpha_bar_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2

    def alpha_bar(self, t):
        """Cosine schedule: alpha_bar(t) = cos^2((t+s)/(1+s) * pi/2) / cos^2(s/(1+s) * pi/2)

        Clamped to [1e-5, 1.0] to avoid division by zero at t=1.
        """
        if isinstance(t, (int, float)):
            val = math.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2 / self._alpha_bar_0
            return max(val, 1e-5)
        val = torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2 / self._alpha_bar_0
        return val.clamp(min=1e-5)

    def diffusion_coeff(self, t):
        """Diffusion coefficient g(t) = sqrt(beta(t)) for the VP-SDE.

        beta(t) = pi * tan(theta(t)) / (1+s),  theta(t) = (t+s)/(1+s) * pi/2.
        """
        if isinstance(t, (int, float)):
            theta = (t + self.s) / (1 + self.s) * math.pi / 2
            beta = math.pi * math.tan(theta) / (1 + self.s)
            return math.sqrt(max(beta, 1e-8))
        theta = (t + self.s) / (1 + self.s) * math.pi / 2
        beta = math.pi * torch.tan(theta) / (1 + self.s)
        return torch.sqrt(beta.clamp(min=1e-8))

    def loss(self, x, cond):
        """Compute training loss (x0-prediction MSE).

        The model predicts the clean data x_0 from the noisy input x_t.

        Args:
            x: (batch, horizon, transition_dim) — normalised trajectories (x_0)
            cond: dict {int: tensor} — conditions (e.g. {0: obs_at_t0})
        """
        batch_size = x.shape[0]
        device = x.device

        # sample continuous time
        t = torch.rand(batch_size, device=device)

        # noise
        eps = torch.randn_like(x)

        # forward diffusion
        abar = self.alpha_bar(t)
        abar = abar.view(batch_size, 1, 1)
        x_t = torch.sqrt(abar) * x + torch.sqrt(1.0 - abar) * eps

        # apply conditioning
        x_t = apply_conditioning(x_t, cond, self.action_dim)

        # model predicts x_0
        x0_pred = self.model(x_t, t)

        # zero out at conditioned positions so they don't contribute to loss
        x0_target = x.clone()
        for pos in cond.keys():
            x0_pred[:, pos, self.action_dim:] = 0.0
            x0_target[:, pos, self.action_dim:] = 0.0

        loss = F.mse_loss(x0_pred, x0_target)

        return loss, {"loss": loss.item()}
