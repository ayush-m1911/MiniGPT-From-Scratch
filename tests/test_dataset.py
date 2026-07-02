from tokenizer.tokenizer import CharacterTokenizer
from data.dataset import ShakespeareDataset

tokenizer = CharacterTokenizer("data/input.txt")

tokens = tokenizer.encode(tokenizer.text)

dataset = ShakespeareDataset(tokens=tokens, block_size=8)

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("Input IDs", x.tolist())
print("Target IDs", y.tolist())

print()

print("Input Text :")
print(tokenizer.decode(x.tolist()))

print()

print("Target Text:")
print(tokenizer.decode(y.tolist()))