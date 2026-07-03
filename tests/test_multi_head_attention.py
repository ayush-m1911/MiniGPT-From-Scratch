import torch

from models.multi_head_attention import MultiHeadAttention
from utils.mask import causal_mask

batch = 2
seq_len = 8
d_model = 384
heads = 6

x = torch.randn(
    batch,
    seq_len,
    d_model
)

mask = causal_mask(
    seq_len,
    x.device
)

model = MultiHeadAttention(
    n_embd=d_model,
    n_head=heads,
    block_size=seq_len
)

output, weights = model(
    x
)

print("Input :", x.shape)
print("Output:", output.shape)
print("Weights:", weights.shape)