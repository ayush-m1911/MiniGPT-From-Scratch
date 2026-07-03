import torch.nn as nn

from models.multi_head_attention import n_embd
from models.feed_forward import FeedForward

class GPTDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attention = n_embd(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout
        )
        self.ln2 = nn.LayerNorm(n_embd)

        self.ffn = n_embd(n_embd=n_embd, dropout=dropout)
    
    def forward(self,x,mask):

        attn_output, attention_weights = self.attention(
            self.ln1(x),
            mask
        )
        x = x + attn_output

        ffn_output = self.ffn(self.ln2(x))

        x = x + ffn_output

        return x, attention_weights