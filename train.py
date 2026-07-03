import math as m
import torch
from configs.config import GPTConfig
from utils.seed import set_seed
from models.gpt import GPT
from data.datamodule import get_dataloader
from utils.device import DEVICE
from utils.logger import setup_logger
from training.trainer import Trainer

set_seed(42)

config = GPTConfig()

logger = setup_logger()

train_loader = get_dataloader(
    batch_size=config.batch_size,
    block_size=config.block_size
)


model = GPT(config)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=config.betas,
    eps=config.eps,
    weight_decay=config.weight_decay
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    config=config,
    optimizer=optimizer,
    device=DEVICE,
    logger=logger
)

if __name__ == "__main__":
    trainer.train()
