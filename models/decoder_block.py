import torch.nn as nn

from models.multi_head_attention import MultiHeadAttention
from models.feed_forward import FeedForward

class GPTDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout
        )
        self.ln2 = nn.LayerNorm(n_embd)

        self.ffn = FeedForward(n_embd=n_embd, dropout=dropout)
    
    def forward(self,x):

        attn_output, attention_weights = self.attention(
            self.ln1(x)
        )
        x = x + attn_output

        ffn_output = self.ffn(self.ln2(x))

        x = x + ffn_output

        return x, attention_weights