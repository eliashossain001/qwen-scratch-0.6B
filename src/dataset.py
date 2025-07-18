# ==================== src/dataset.py ====================
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=2048):
        # Load entire text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Tokenize into IDs
        tokens = tokenizer.encode(text).ids
        # Create sliding windows of length seq_len
        self.inputs = []
        for i in range(0, len(tokens) - seq_len):
            chunk = tokens[i : i + seq_len]
            self.inputs.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        # For language modeling, input and target are the same sequence.
        return x, x


def get_dataloader(file_path, seq_len, batch_size):
    # Ensure tokenizer files exist
    vocab_path = os.path.join("data", "tokenizer", "vocab.json")
    merges_path = os.path.join("data", "tokenizer", "merges.txt")
    if not os.path.isfile(vocab_path) or not os.path.isfile(merges_path):
        raise FileNotFoundError(f"Tokenizer files not found at {vocab_path} and {merges_path}")

    # Load the trained BPE tokenizer
    tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

    # Create dataset and dataloader
    ds = TextDataset(file_path, tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
