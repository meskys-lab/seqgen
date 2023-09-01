from pathlib import Path
import pickle
from typing import Any, Union

import pandas as pd
import torch
from esm import FastaBatchedDataset, BatchConverter, Alphabet

from seqgen.model.utils import get_backbone


def get_train_dataset(data: pd.DataFrame) -> FastaBatchedDataset:
    return FastaBatchedDataset(sequence_labels=data.id, sequence_strs=data.sequence)



def parse_fasta(path: Union[Path, str]) -> pd.DataFrame:
    seq_data = pd.read_csv(path, lineterminator=">", header=None)
    seq_data = seq_data[["title", "sequence", "empty_col"]] = seq_data[0].str.split("\n", expand=True)
    seq_data = seq_data[["title", "sequence"]]
    return seq_data



class Converter():

    def __init__(self, name:str) -> None:
        self.backbone = get_backbone(name).eval().cuda()
        self.collate_fn = BatchConverter(Alphabet.from_architecture("ESM-1b"))



    def __call__(self, item:list) -> Any:
        with torch.no_grad():
            item = self.collate_fn([item])
            output = self.backbone(item[2].cuda(), repr_layers=[self.backbone.num_layers])
            rep = output['representations'][self.backbone   .num_layers].detach().cpu().numpy()
        return item[0], item[1], item[2].numpy().astype("uint8").squeeze(), rep.astype("float16").squeeze()
        