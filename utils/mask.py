import torch


def causal_mask(seq_len, device):

    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device)
    )

    return mask.unsqueeze(0).unsqueeze(0)