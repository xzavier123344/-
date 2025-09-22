import os
import pickle
from collections import Counter
import config

def build_vocab(file_path):
    words = []
    tags = []

    # 读取训练数据
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line.split()) != 2:
                continue  # 忽略格式异常行
            word, tag = line.split()
            words.append(word)
            tags.append(tag)

    # 构建词表和标签表
    word_counter = Counter(words)
    vocab = [config.PAD_TOKEN, config.UNK_TOKEN] + list(word_counter.keys())
    word2id = {word: idx for idx, word in enumerate(vocab)}

    tag_set = sorted(set(tags))
    tag2id = {tag: idx for idx, tag in enumerate(tag_set)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(config.VOCAB_PATH), exist_ok=True)

    # 保存为 pickle
    with open(config.VOCAB_PATH, 'wb') as f:
        pickle.dump((word2id, tag2id, id2tag), f)

    print(f"✅ 字典保存成功：{config.VOCAB_PATH}")
    print(f"📦 共 {len(word2id)} 个词，{len(tag2id)} 个标签")

if __name__ == "__main__":
    build_vocab(config.TRAINING_DATA_PATH)
