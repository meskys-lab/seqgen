from ast import Tuple
import pickle
import torch
from torch.utils.data import Dataset


class SeqRepresentationDataset(Dataset):
    def __init__(self, path: str, max_length: int = 160):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        id, seq, aa_idx, rep = self.data[idx]
        rep = (rep - rep.mean()) / rep.std()
        _rep = torch.tensor(rep, device="cuda", dtype=torch.float)[: self.max_length]
        rep = torch.zeros(
            self.max_length, rep.shape[-1], device="cuda", dtype=torch.float
        )
        rep[: len(_rep), :] = _rep
        _aa_idx = torch.tensor(aa_idx, device="cuda", dtype=torch.int)[
            : self.max_length
        ]
        aa_idx = torch.ones(self.max_length, device="cuda", dtype=torch.int)
        aa_idx[: len(_aa_idx)] = _aa_idx
        return rep, aa_idx
