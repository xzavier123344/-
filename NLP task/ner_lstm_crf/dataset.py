import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, file_path, word2id, tag2id):
        self.sentences = []
        self.labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        self.sentences.append([word2id.get(w, word2id['<UNK>']) for w in words])
                        self.labels.append([tag2id[t] for t in tags])
                        words, tags = [], []
                else:
                    word, tag = line.split()
                    words.append(word)
                    tags.append(tag)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx])

def collate_fn(batch):
    sentences, labels = zip(*batch)
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)

    padded_sents = torch.full((len(sentences), max_len), fill_value=0, dtype=torch.long)
    padded_labels = torch.full((len(sentences), max_len), fill_value=0, dtype=torch.long)
    masks = torch.zeros(len(sentences), max_len, dtype=torch.bool)

    for i, (s, l) in enumerate(zip(sentences, labels)):
        padded_sents[i, :len(s)] = s
        padded_labels[i, :len(l)] = l
        masks[i, :len(s)] = 1

    return padded_sents, padded_labels, masks