import torch
from torch.utils.data import DataLoader, random_split

from tokenizer.tokenizer import CharacterTokenizer
from data.dataset import ShakespeareDataset


def get_dataloaders(
    batch_size,
    block_size,
    train_ratio=0.9
):
    tokenizer = CharacterTokenizer("data/input.txt")

    tokens = tokenizer.encode(tokenizer.text)

    dataset = ShakespeareDataset(tokens, block_size)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader