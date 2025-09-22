import json
import torch
from torch.utils.data import Dataset
import re
from collections import Counter

# 标签映射
LABEL2ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

def tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def load_snli_data(path, max_samples=None):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            label = obj['gold_label']
            if label not in LABEL2ID:
                continue  # 跳过无效样本
            premise = obj['sentence1']
            hypothesis = obj['sentence2']
            data.append((premise, hypothesis, LABEL2ID[label]))
            if max_samples and len(data) >= max_samples:
                break
    return data  # 返回 (premise, hypothesis, label) 列表

def build_vocab(dataset, max_vocab_size=20000, min_freq=1):
    counter = Counter()
    for s1, s2, _ in dataset:
        counter.update(tokenize(s1))
        counter.update(tokenize(s2))
    most_common = [w for w, c in counter.items() if c >= min_freq][:max_vocab_size - 2]
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(most_common, 2):
        word2idx[word] = i
    return word2idx

def encode(text, word2idx, max_len):
    tokens = tokenize(text)
    ids = [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]
    if len(ids) < max_len:
        ids += [word2idx["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids)

class SNLIDataset(Dataset):
    def __init__(self, data, word2idx, max_len=50):
        self.data = data
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise, hypothesis, label = self.data[idx]
        p_ids = encode(premise, self.word2idx, self.max_len)
        h_ids = encode(hypothesis, self.word2idx, self.max_len)
        return p_ids, h_ids, torch.tensor(label)