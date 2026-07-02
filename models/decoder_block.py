import torch.nn as nn

from models.multi_head_attention import MultiHeadCausalAttention
from models.feed_forward import FeedForward

class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attention = MultiHeadCausalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model=d_model, dropout=dropout)
    
    def forward(self,x,mask):

        attn_output, attention_weights = self.attention(
            self.ln1(x),
            mask
        )
        x = x + attn_output

        ffn_output = self.ffn(self.ln2(x))

        x = x + ffn_output

        return x, attention_weights