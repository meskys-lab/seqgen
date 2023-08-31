import numpy as np

from seqgen.data.dataloader import ALPHABET


def get_recovery_rate(logits, aa_idx, padding_mask):
     recovery_rate = (logits.argmax(-1) == aa_idx).float().detach().cpu().numpy()
     mask = 1-padding_mask.detach().cpu().numpy()
     recovery_rate_av = (recovery_rate * mask).sum(-1) / np.maximum(1, mask.sum(-1))
     return recovery_rate_av

def print_logits_to_seq(logits):
    aa_index = logits.argmax(-1).cpu().detach()
    for i in range(len(aa_index)):
        print(''.join([ALPHABET.get_tok(a) for a in aa_index[i]]))