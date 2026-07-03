from tests.test_causal_attention import seq_len
import torch
import torch.nn as nn

from models.embeddings import GPTEmbedding
from models.decoder_block import GPTDecoderBlock
from utils.mask import causal_mask

class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
    "Embedding dimension must be divisible by number of heads."
)
        self.config = config

        self.embedding = GPTEmbedding(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            block_size=config.block_size,
            dropout=config.dropout
        )
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                dropout=config.dropout
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self,tokens):
        batch_size, seq_len = tokens.shape

        mask = causal_mask(seq_len, tokens.device)

        x = self.embedding(tokens)

        attention_maps = []

        for block in self.blocks:
            x, attention = block(
                x, mask
            )
            attention_maps.append(attention)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits, attention_maps