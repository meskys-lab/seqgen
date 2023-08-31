from functools import partial
from esm import Alphabet
from fastai.data.core import DataLoaders
from torch.utils.data import DataLoader

from seqgen.data.dataset import SeqRepresentationDataset

ALPHABET = Alphabet.from_architecture("ESM-1b")
from torch.nn.utils.rnn import pad_sequence



def collate(batch, max_length = 160):
    rep = pad_sequence([e[0][:max_length] for e in batch], batch_first=True, padding_value=0.0)
    seqs = pad_sequence([e[1][:max_length] for e in batch], batch_first=True, padding_value=1.0)
    return rep , seqs


def get_dataloader(path: str, batch_size=8, max_length = 160) -> DataLoader:
    dataset = SeqRepresentationDataset(path, max_length=max_length)
    return DataLoader(dataset, collate_fn=partial(collate, max_length=max_length), batch_size=batch_size, shuffle=True)


def get_dataloaders(train_path: str, val_path: str, batch_size=8, max_length = 160) -> DataLoaders:
    train_dataloader = get_dataloader(train_path, batch_size, max_length)
    val_dataloader = get_dataloader(val_path, batch_size, max_length)

    return DataLoaders(train_dataloader, val_dataloader, device="cuda")
