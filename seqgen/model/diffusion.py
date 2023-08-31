# Adopted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

from random import random
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from einops import reduce
from tqdm import tqdm
from seqgen.model.utils import (
    linear_beta_schedule,
    cosine_beta_schedule,
    default,
    extract,
    identity,
    l2_loss_with_norm_penalty,
)


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        head,
        *,
        sample_shape=(160, 33),
        timesteps=100,
        sampling_timesteps=None,
        loss_type="l1",
        objective="pred_noise",
        beta_schedule="cosine",
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    ):
        super().__init__()

        self.model = model
        self.head = head
        self.self_condition = self.model.self_condition

        self.sample_shape = sample_shape

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start)"

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting

        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, rep_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, rep_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, rep_t.shape) * rep_t
        )
        posterior_variance = extract(self.posterior_variance, t, rep_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, rep_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self, x, t, padding_mask, x_self_cond=None, clip_x_start=False
    ):
        model_output = self.model(x, t, x_self_cond, padding_mask=padding_mask)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self, rep_t, t, padding_mask, x_self_cond=None, clip_denoised=True
    ):
        preds = self.model_predictions(
            rep_t,
            t,
            padding_mask=padding_mask,
            x_self_cond=x_self_cond,
        )
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, rep_t=rep_t, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, rep_t, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *rep_t.shape, rep_t.device
        batched_times = torch.full(
            (rep_t.shape[0],), t, device=device, dtype=torch.long
        )
        padding_mask = torch.zeros(*rep_t.shape[:2], dtype=torch.bool).to(rep_t.device)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            rep_t=rep_t,
            t=batched_times,
            padding_mask=padding_mask,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(rep_t) if t > 0 else 0.0  # no noise if t == 0
        rep = model_mean + (0.5 * model_log_variance).exp() * noise
        return rep, x_start

    @torch.no_grad()
    def sample(self, batch_size=16, shape=None):
        device = self.betas.device

        rep = torch.randn((shape or batch_size, *self.sample_shape), device=device)

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            rep, x_start = self.p_sample(rep, t, self_cond)

        # img = unnormalize_to_zero_to_one(img)
        return rep

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return partial(F.l1_loss, reduction="none")
        elif self.loss_type == "l2":
            return l2_loss_with_norm_penalty
        elif self.loss_type == "cos":
            return F.cosine_similarity
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def get_loss(self, x_start, t, padding_mask, noise=None, aa_idx=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x, t, padding_mask=padding_mask
                ).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond, padding_mask=padding_mask)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = self.loss_fn(model_out, target)
        loss = reduce(loss, "b ... -> b (...)", "mean")

        if aa_idx is not None:
            denoised = self.predict_start_from_noise(x, t, model_out)
            denoised.clamp_(-1.0, 1.0)
            denoised = (
                extract(self.posterior_mean_coef1, t, x.shape) * denoised
                + extract(self.posterior_mean_coef2, t, x.shape) * x
            )

            logits = self.head(denoised.detach())
            aa_loss = self.head.get_loss(logits, aa_idx)
            loss = loss + (aa_loss * 0.1)

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, sample, padding_mask, *args, **kwargs):
        device = sample.device
        b = sample.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.get_loss(sample, t, padding_mask=padding_mask, *args, **kwargs)
