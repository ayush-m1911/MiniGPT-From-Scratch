import torch

from models.embeddings import GPTEmbedding

vocab_size = 65
d_model = 384
block_size = 128

model = GPTEmbedding(
    vocab_size=vocab_size,
    d_model=d_model,
    block_size=block_size
)

tokens = torch.randint(
    0,
    vocab_size,
    (2, 8)
)

output = model(tokens)

print("Input Shape :", tokens.shape)
print("Output Shape:", output.shape)