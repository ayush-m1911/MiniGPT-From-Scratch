from torch.utils.data import DataLoader
from tokenizer.tokenizer import CharacterTokenizer
from data.dataset import ShakespeareDataset


def get_dataloader(batch_size, block_size):
    tokenizer = CharacterTokenizer("data/input.txt")

    tokens = tokenizer.encode(tokenizer.text)

    dataset = ShakespeareDataset(tokens, block_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )