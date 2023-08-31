from pathlib import Path
import numpy as np
from torch.optim import Adam

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
import torch
from seqgen.model.metrics import get_recovery_rate, print_logits_to_seq
from seqgen.model.utils import cycle, exists, extract, num_to_groups
from seqgen.data.dataloader import ALPHABET


class Learner(object):
    def __init__(
        self,
        diffusion_model,
        dataloader,
        val_dataloader,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=10000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=250,
        num_samples=4,
        results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches, mixed_precision="fp16" if fp16 else "no"
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model
        self.head = diffusion_model.head

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        dl = self.accelerator.prepare(dataloader)
        self.dl = cycle(dl)
        self.val_dataloader = val_dataloader

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @torch.no_grad()
    def eval(self, device):
        recs, denoise_losses, aa_losses = [], [], []
        for batch in self.val_dataloader:
            rep = batch[0].to(device)
            aa_idx = batch[1].to(device)
            padding_mask = aa_idx.eq(ALPHABET.padding_idx)

            noise = torch.randn_like(rep)
            t = torch.randint(
                0, self.model.num_timesteps, (rep.shape[0],), device=rep.device
            ).long()
            x = self.model.q_sample(x_start=rep, t=t, noise=noise)

            model_out = self.model.model(
                x, t, x_self_cond=None, padding_mask=padding_mask
            )
            denoised = self.model.predict_start_from_noise(x, t, model_out)

            denoised.clamp_(-1.0, 1.0)
            denoised = (
                extract(self.model.posterior_mean_coef1, t, x.shape) * denoised
                + extract(self.model.posterior_mean_coef2, t, x.shape) * x
            )

            logits = self.model.head(denoised)

            denoise_losses.append(
                self.model.loss_fn(model_out, noise).cpu().detach().numpy()
            )
            aa_losses.append(
                self.model.head.get_loss(logits, aa_idx).cpu().detach().numpy()
            )

            recs.append(get_recovery_rate(logits, aa_idx, padding_mask))

        recsm = np.concatenate(recs).mean()
        dem = np.concatenate(denoise_losses).mean()
        aam = np.asarray(aa_losses).mean()
        print(
            f"Recovery rate:{recsm:.2f}| Denoising loss: {dem:.2f}| Cross entrophy loss: {aam:.2f}"
        )

    def save(self, name):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }

        path = str(self.results_folder / f"{name}.pt")
        torch.save(data, path)
        return path

    def load(self, name):
        accelerator = self.accelerator
        device = accelerator.device

        if name.endswith(".pt"):
            path = name
        else:
            path = str(self.results_folder / f"{name}.pt")
        data = torch.load(
            path, map_location=device
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    rep = data[0].to(device)
                    aa_idx = data[1].to(device)
                    padding_mask = aa_idx.eq(ALPHABET.padding_idx)

                    with self.accelerator.autocast():
                        loss = self.model(rep, padding_mask=padding_mask, aa_idx=aa_idx)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            self.eval(device)
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            denoised_rep_batches = list(
                                map(
                                    lambda n: self.ema.ema_model.sample(batch_size=n),
                                    batches,
                                )
                            )
                            for batch in denoised_rep_batches:
                                logits = self.head(batch)
                                print_logits_to_seq(logits)

                pbar.update(1)

        accelerator.print("training complete")
