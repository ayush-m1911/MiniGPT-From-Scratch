import torch
import torch.nn as nn

from models.attention import CausalSelfAttention

class MultiHeadCausalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(
            d_model,
            3*d_model
        )

        self.proj = nn.Linear(d_model, d_model)

        self.attention = CausalSelfAttention(dropout)

        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x,
        mask
    ):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)

        Q, K, V = qkv.chunk(3,dim = -1)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        output, attention = self.attention(Q,K,V,mask)

        output = output.transpose(1,2).contiguous()
        output = output.view(batch_size,seq_len,self.d_model)

        output = self.proj(output)

        output = self.dropout(output)

        return output, attention