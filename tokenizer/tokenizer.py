from sympy import Idx
from pathlib import Path

class CharacterTokenizer:
    def __init__(self,file_path: str):

        self.text = Path(file_path).read_text(encoding="utf-8")

        chars = sorted(list(set(self.text)))

        self.vocab_size = len(chars)
        
        self.stoi = {
            ch : idx
            for idx,ch in enumerate(chars)
        }

        self.itos = {
            idx: ch
            for idx,ch in enumerate(chars)
        }
    
    def encode(self, text: str):
        
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):

        return "".join(
            self.itos[t]
            for t in tokens
        )