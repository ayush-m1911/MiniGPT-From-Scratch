import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embd, d_ff=None, dropout=0.1):
        
        super().__init__()
        
        if d_ff is None:
            d_ff = 4*n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)