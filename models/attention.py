import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self,dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(self,Q,K,V, mask=None):
        d_k = Q.size(-1)

        scores = torch.matmul(Q,K.transpose(-2,-1))

        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention,V)

        return output, attention