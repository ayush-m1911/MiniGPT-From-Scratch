import torch

from models.decoder_block import GPTDecoderBlock
from utils.mask import causal_mask

batch = 2
seq_len = 16
d_model = 384
num_heads = 6

x = torch.randn(
    batch,
    seq_len,
    d_model
)

mask = causal_mask(
    seq_len,
    x.device
)

block = GPTDecoderBlock(
    n_embd=d_model,
    n_head=num_heads
)

output, weights = block(
    x,
    mask
)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)
print("Attention Shape:", weights.shape)