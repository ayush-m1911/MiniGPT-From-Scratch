import math as m
import torch
from configs.config import GPTConfig
from utils.seed import set_seed
from models.gpt import GPT
from data.datamodule import get_dataloader
from utils.device import DEVICE
from utils.logger import setup_logger
from training.trainer import Trainer
from utils.model_utils import count_parameters

def main():

    set_seed(42)

    config = GPTConfig()

    logger = setup_logger()

    logger.info("=" * 60)
    logger.info("GPT From Scratch")
    logger.info("=" * 60)

    logger.info(f"Using Device : {DEVICE}")

    train_loader = get_dataloader(
        batch_size=config.batch_size,
        block_size=config.block_size
    )

    model = GPT(config)

    logger.info(
        f"Parameters : {count_parameters(model):,}"
    )

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

    trainer.train()


if __name__ == "__main__":
    main()