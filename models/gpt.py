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
                block_size=config.block_size,
                dropout=config.dropout
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head_weight = self.embedding.token_embedding.weight

        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self,tokens):
        batch_size, seq_len = tokens.shape

        x = self.embedding(tokens)

        attention_maps = []

        for block in self.blocks:
            x, attention = block(x)
            attention_maps.append(attention)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits, attention_maps