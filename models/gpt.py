from tests.test_causal_attention import seq_len
import torch
import torch.nn as nn

from models.embeddings import GPTEmbedding
from models.decoder_block import GPTDecoderBlock
from utils.mask import causal_mask

class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config

        self.embedding = GPTEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            block_size=config.block_size,
            dropout=config.dropout
        )
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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