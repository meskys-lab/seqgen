import argparse
import logging
from argparse import ArgumentParser
import pickle

import pandas as pd
import torch
from seqgen.data.data import Converter, get_train_dataset




def parse_train_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="esm2_t6_8M_UR50D", help='ESM model name')
    parser.add_argument('--esm_model_hub', type=str, default='/mnt/shared/models/esm/weights/hub',
                        help='Path where ESM models are downloaded')
    parser.add_argument('--input_file', required=True, type=str, help='Full path to file which contains sequences in csv format (id, sequence)')
    parser.add_argument('--output_path', type=str, default="example/dataset.p",
                        help='Full path where to store representations of sequences in fasta file')
    args = parser.parse_args()
    return args


def convert(args: argparse.Namespace):
    logging.info(f"Starting predicting using model {args.model}")

    c = Converter(args.model)
    df = pd.read_csv(args.input_file)
    dataset = get_train_dataset(df)

    data = []
    for b in dataset:
        item = c(b)
        data.append(item)

    with open(args.output_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_train_args()
    torch.hub.set_dir(args.esm_model_hub)

    convert(args)
