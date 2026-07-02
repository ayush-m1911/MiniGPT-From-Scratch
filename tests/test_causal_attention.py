import torch

from models.attention import CausalSelfAttention
from utils.mask import causal_mask

batch = 2
heads = 4
seq_len = 8
head_dim = 16

Q = torch.randn(batch, heads, seq_len, head_dim)
K = torch.randn(batch, heads, seq_len, head_dim)
V = torch.randn(batch, heads, seq_len, head_dim)

mask = causal_mask(seq_len, Q.device)

attention = CausalSelfAttention()

output, weights = attention(Q, K, V, mask)

print("Output Shape :", output.shape)
print("Weights Shape:", weights.shape)