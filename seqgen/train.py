import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn

from seqgen.data.dataloader import get_dataloaders
from seqgen.learner import Learner
from seqgen.model.diffusion import Diffusion
from seqgen.model.modules.aa_head import AaHead
from seqgen.model.modules.unet import Unet


def parse_train_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to a train pickle file"
    )
    parser.add_argument(
        "--val_data", type=str, default=None, help="Path to a val pickle file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Path to folder with trained model",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--unet_dims", type=int, default=32, help="Unet model dimension"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=160, help="Maximum sequence length allowed"
    )
    parser.add_argument("--emb_dim", type=int, default=320, help="Embedding dimension")
    parser.add_argument(
        "--timesteps", type=int, default=100, help="Time steps of diffusion process"
    )
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=10000,
        help="Time steps of diffusion process",
    )
    parser.add_argument(
        "--backbone", type=str, default="esm2_t6_8M_UR50D", help="Type of the backbone"
    )
    parser.add_argument(
        "--model", type=str, default="esm_rep", help="Type of the model"
    )
    parser.add_argument(
        "--esm_model_hub",
        type=str,
        default="/mnt/shared/models/esm/weights/hub",
        help="Path where ESM models are downloaded",
    )

    args = parser.parse_args()
    return args


def train(args):
    logging.info(f"Starting training with these parameters: {args}")

    dataloaders = get_dataloaders(
        train_path=args.train_data, val_path=args.val_data, batch_size=args.batch_size
    )

    learner = get_learner(dataloaders, args)

    name = get_model_name(learner.model)

    save_config(args.model_dir, name)

    learner.train()

    path = learner.save(name)

    logging.info(f"Model has been save: {path}")


def get_learner(dataloaders, args):
    diff_model = get_diff_model(args)
    learner = Learner(
        diff_model,
        dataloader=dataloaders[0],
        val_dataloader=dataloaders[1],
        train_num_steps=args.train_num_steps,
        results_folder=args.model_dir,
    )
    return learner


def get_diff_model(args):
    denoiser = Unet(dim=args.unet_dims)
    head = AaHead(name=args.backbone)
    diff_model = Diffusion(
        denoiser,
        head=head,
        sample_shape=(args.max_seq_len, args.emb_dim),
        timesteps=args.timesteps,
    ).cuda()

    return diff_model


def get_model_name(model: nn.Module) -> str:
    name = model.__class__.__name__
    model_name = f"{name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    return model_name


def save_config(model_dir, name: str) -> None:
    config_file = (Path(model_dir) / name).with_suffix(".yaml")
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, "w") as file:
        yaml.dump(vars(args), file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_train_args()

    torch.hub.set_dir(args.esm_model_hub)

    train(args)
