import torch
from torch.utils.data import Dataset
import re
from collections import Counter
import csv

def load_data(filepath, has_labels=True):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            texts.append(row['Phrase'])
            if has_labels:
                labels.append(int(row['Sentiment']))
    return texts, labels if has_labels else texts

def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

def build_vocab(texts, max_vocab_size=20000, min_freq=1):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    most_common = [w for w, c in counter.items() if c >= min_freq][:max_vocab_size - 2]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(most_common, start=2):
        vocab[word] = i
    return vocab

def encode_sentence(text, word2idx, max_len):
    tokens = tokenize(text)
    ids = [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]
    if len(ids) < max_len:
        ids += [word2idx["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids)

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=50):
        self.data = [encode_sentence(t, word2idx, max_len) for t in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
