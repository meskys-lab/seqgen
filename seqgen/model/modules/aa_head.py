import logging
import math
from einops import rearrange

import numpy as np

from torch import nn, Tensor
import torch
from seqgen.data.dataloader import ALPHABET
from seqgen.model.utils import get_backbone
import torch.nn.functional as F


class AaHead(nn.Module):
    def __init__(self, name: str, pad_weight=0.1) -> None:
        super(AaHead, self).__init__()
        self.layers = get_backbone(name).cuda().lm_head
        pad_weight = (np.asarray(ALPHABET.all_toks) == "<pad>") * (1 - pad_weight)
        class_weights = np.ones(len(ALPHABET.all_toks)) - pad_weight
        self.class_weights = torch.tensor(class_weights, device="cuda").float()

    def forward(self, rep: Tensor, t: Tensor=None, x_self_cond: Tensor=None, padding_mask: Tensor=None) -> Tensor:
        logits = self.layers(rep)
        return logits

    def get_loss(self, logits: Tensor, aa_idx: Tensor) -> Tensor:
        aa_loss = F.cross_entropy(
            logits.transpose(1, 2),
            aa_idx.long(),
            label_smoothing=0.9,
            weight=self.class_weights,
        )

        return aa_loss
