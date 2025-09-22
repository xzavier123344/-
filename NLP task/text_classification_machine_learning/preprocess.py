import numpy as np
import re
from collections import defaultdict
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
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def build_vocab(texts, ngram=1, max_features=None):
    vocab = defaultdict(int)
    for text in texts:
        tokens = tokenize(text)
        ngrams = zip(*[tokens[i:] for i in range(ngram)])
        for gram in ngrams:
            vocab[' '.join(gram)] += 1

    # 词频排序并保留 max_features
    sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])
    if max_features:
        sorted_vocab = sorted_vocab[:max_features]
    word2idx = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    return word2idx

def vectorize(texts, vocab, ngram=1):
    vectors = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        ngrams = zip(*[tokens[j:] for j in range(ngram)])
        for gram in ngrams:
            key = ' '.join(gram)
            if key in vocab:
                vectors[i][vocab[key]] += 1
    return vectors
