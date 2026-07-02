import torch

from configs.config import GPTConfig
from models.gpt import GPT

config = GPTConfig()

model = GPT(config)

tokens = torch.randint(
    0,
    config.vocab_size,
    (2, 16)
)

logits, attention = model(tokens)

print("Input Shape :", tokens.shape)
print("Logits Shape:", logits.shape)
print("Decoder Layers:", len(attention))
print("One Attention Shape:", attention[0].shape)