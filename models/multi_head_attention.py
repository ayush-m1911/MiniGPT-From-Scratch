import torch
import torch.nn as nn

from models.attention import CausalSelfAttention

class n_embd(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(
            n_embd,
            3*n_embd
        )

        self.proj = nn.Linear(n_embd, n_embd)

        self.attention = CausalSelfAttention(dropout)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                .view(1, 1, block_size, block_size)
        )
    
    def forward(
        self,
        x
    ):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)

        Q, K, V = qkv.chunk(3,dim = -1)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        mask = self.mask[:, :, :seq_len, :seq_len]

        output, attention = self.attention(
            Q,
            K,
            V,
            mask
        )
        output = output.transpose(1,2).contiguous()
        output = output.view(batch_size,seq_len,self.d_model)

        output = self.proj(output)

        output = self.dropout(output)

        return output, attention