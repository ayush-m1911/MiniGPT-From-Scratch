from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    expansion_factor: int = 4
    dropout: float = 0.2

    batch_size: int = 64
    epochs: int = 10

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    warmup_steps: int = 500

    seed: int = 42

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            "Embedding dimension must be divisible by number of heads."

        assert self.block_size > 0
        assert self.vocab_size > 0
        assert self.dropout >= 0.0
        assert self.dropout <= 1.0