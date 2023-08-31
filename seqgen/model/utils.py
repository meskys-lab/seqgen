import logging
import re
import math

import torch
from esm.model.esm2 import ESM2
from torch import nn

from seqgen.data.dataloader import ALPHABET


def get_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if n.startswith("backbone"):
            p.requires_grad = False

    logging.info(f"Model has {get_trainable_parameters(model)} parameters")


def freeze_embeddings(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "backbone.embed_tokens" in n:
            p.requires_grad = False

    logging.info(f"Model has {get_trainable_parameters(model)} parameters")


def unfreeze(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = True

    logging.info(f"Model has {get_trainable_parameters(model)} parameters")


def unfreeze_head(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if n.startswith("head"):
            p.requires_grad = True

    logging.info(f"Model has {get_trainable_parameters(model)} parameters")


def upgrade_state_dict(state_dict: dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


def get_backbone(name: str) -> nn.Module:
    path = f"{torch.hub.get_dir()}/checkpoints/{name}.pt"

    logging.info(f"Loading model from {path}")

    model_data = torch.load(path, map_location="cpu")
    cfg = model_data["cfg"]["model"]

    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)

    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=ALPHABET,
        token_dropout=cfg.token_dropout,
    )

    model.load_state_dict(state_dict, strict=False)

    trainable_params = get_trainable_parameters(model)

    logging.info(f"Model {name} loaded. It has {trainable_params} trainable parameters")

    return model


def l2_norm_tns(tns):
    return tns / tns.norm(dim=-1, p=2, keepdim=True)


def l2_norm_loss(pred, true):
    pred_normed = l2_norm_tns(pred)
    true_normed = l2_norm_tns(true)
    l2_loss = F.mse_loss(pred_normed, true_normed, reduction="none").sum(-1)
    return l2_loss


def l2_loss_with_norm_penalty(pred, true):
    l2_loss = l2_norm_loss(pred, true)
    norm_penalty = torch.abs(pred.norm(dim=-1, p=2) - 1)
    loss = l2_loss + 0.02 * norm_penalty
    return loss


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    assert t.shape[0] == x_shape[0], "batch size of t and x_shape must match"
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)



def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
