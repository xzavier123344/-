import numpy as np
import torch

def load_glove_embeddings(glove_path, word2idx, embedding_dim=100):
    print("🔍 加载 GloVe 词向量中...")
    embeddings = np.random.uniform(-0.05, 0.05, (len(word2idx), embedding_dim)).astype(np.float32)
    found = 0

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=np.float32)
            if word in word2idx:
                embeddings[word2idx[word]] = vector
                found += 1

    print(f"✅ GloVe 加载完成！找到 {found} / {len(word2idx)} 个词的预训练向量。")
    return torch.tensor(embeddings)