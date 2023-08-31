import math
from einops import rearrange


from torch import nn, Tensor
import torch

from seqgen.model.utils import exists


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


class TransfomerDenoiser(nn.Module):
    def __init__(self, backbone: nn.Module, dim=32) -> None:
        super(TransfomerDenoiser, self).__init__()
        self.layers = backbone.layers
        self.emb_layer_norm_after = backbone.emb_layer_norm_after
        self.self_condition = True

        sinu_pos_emb = SinusoidalPosEmb(dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 320 * 2)
        )

        self.reduce_dim = nn.Sequential(
            nn.Linear(320 * 2, 320), nn.GELU(), nn.Linear(320, 320)
        )

    def forward(self, rep: Tensor, t, x_self_cond, padding_mask: Tensor) -> Tensor:
        x = rep.transpose(0, 1)

        if self.self_condition:
            if x_self_cond is not None:
                x_self_cond = x_self_cond.transpose(0, 1)
            else:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=2)
            x = self.reduce_dim(x)

        if exists(self.time_mlp) and exists(t):
            time_emb = self.time_mlp(t)
            time_emb = rearrange(time_emb, "b c -> 1 b c")
            scale, shift = time_emb.chunk(2, dim=2)

        x = x * (scale + 1) + shift

        for layer_idx, layer in enumerate(self.layers):
            x, _ = layer(x, self_attn_padding_mask=padding_mask)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        return x
