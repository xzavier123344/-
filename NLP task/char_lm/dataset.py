import torch
from torch.utils.data import Dataset

def load_poetry_dataset(file_path, seq_len=100):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = CharDataset(text, seq_len=seq_len)
    return dataset

class CharDataset(Dataset):
    def __init__(self, text, seq_len=100):
        self.seq_len = seq_len

        # 构建字符集
        chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)

        # 全部文本转为索引序列
        self.data = [self.char2idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)