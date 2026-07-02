from base64 import decode
from tokenizer.tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer("data/input.txt")
print("Vocabulary Size:", tokenizer.vocab_size)

sample = "To be"

tokens = tokenizer.encode(sample)

decoded = tokenizer.decode(tokens)

print("Original:", sample)
print("Encoded :", tokens)
print("Decoded :", decoded)

print(f"Dataset length : {len(tokenizer.text):,}")
print(f"Vocabulary size: {tokenizer.vocab_size}")

print("\nFirst 200 characters:\n")
print(tokenizer.text[:200])