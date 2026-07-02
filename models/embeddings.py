import torch 
import torch.nn as nn

class GPTEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, block_size: int, dropout: float = 0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )
        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,tokens):

        batch_size, seq_len = tokens.shape

        positions = torch.arange(
            seq_len,
            device = tokens.device
        )

        positions = positions.unsqueeze(0).expand(batch_size,seq_len)

        token_embeddings = self.token_embedding(tokens)

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        return self.dropout(x)
