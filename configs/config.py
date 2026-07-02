from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65

    block_size: int = 128

    d_model: int = 384

    num_heads: int = 6
    
    num_layers: int = 6

    dropout: float = 0.2