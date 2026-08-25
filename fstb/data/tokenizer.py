import torch
from typing import List

class FSTBTokenizer:
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        # ID space from 4 to vocab_size - 1

    def _hash_word(self, word: str) -> int:
        hash_val = 5381
        for char in word:
            hash_val = ((hash_val << 5) + hash_val) + ord(char)
        return (hash_val % (self.vocab_size - 4)) + 4

    def encode(self, text: str, max_len: int, add_bos: bool = True, add_eos: bool = True) -> torch.Tensor:
        words = text.split()
        token_ids = []
        if add_bos:
            token_ids.append(self.BOS)
            
        for w in words:
            token_ids.append(self._hash_word(w))
            
        if add_eos:
            token_ids.append(self.EOS)

        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            if add_eos: 
                token_ids[-1] = self.EOS
        else:
            token_ids = token_ids + [self.PAD] * (max_len - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        words = []
        for i in ids.tolist():
            if i == self.PAD:
                continue
            elif i == self.BOS:
                words.append("<BOS>")
            elif i == self.EOS:
                words.append("<EOS>")
            elif i == self.UNK:
                words.append("<UNK>")
            else:
                words.append(f"<TOKEN_{i}>")
        return " ".join(words)

    def batch_encode(self, texts: List[str], max_len: int) -> torch.Tensor:
        encoded = [self.encode(text, max_len) for text in texts]
        return torch.stack(encoded)
