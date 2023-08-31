import argparse
import csv
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import torch
import yaml

from seqgen.data.dataloader import ALPHABET
from seqgen.learner import Learner

from seqgen.train import get_diff_model


def parse_train_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--esm_model_hub', type=str, default='/mnt/shared/models/esm/weights/hub',
                        help='Path where ESM models are downloaded')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output', type=str, default="example/predictions.fasta",
                        help='Full path where to store generated sequences')
    args = parser.parse_args()
    return args


def predict(args: argparse.Namespace):
    logging.info(f"Starting predicting using model {args.model}")

    with open(Path(args.model).with_suffix(".yaml")) as f:
        config = yaml.safe_load(f)

    train_args = Namespace(**config)
    diff_model = get_diff_model(train_args)
    learner = Learner(diff_model, dataloader=None, val_dataloader=None)

    learner.load(args.model)

    p = learner.model.sample(args.batch_size)

    seqs, ids = [], []
    logits = learner.head(p)
    aa_index = logits.argmax(-1).cpu().detach()
    for i in range(len(aa_index)):
        seqs.append(''.join([ALPHABET.get_tok(a) for a in aa_index[i]]))
        ids.append(f">{i}")
    
    results_df = pd.DataFrame({"id":ids,"sequence":seqs})
    save_output(args, results_df)


def save_output(args: argparse.Namespace, results_df: pd.DataFrame) -> None:
    logging.info(f"Prediction will be save in {args.output}")
    results_df.to_csv(args.output, sep='\n', header=False, index=False, quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_train_args()
    torch.hub.set_dir(args.esm_model_hub)

    predict(args)
