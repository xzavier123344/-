import torch
import torch.nn.functional as F
from model import ESIM
from utils import set_seed
from dataset import build_vocab, load_snli_data
import config

# 标签映射（根据 SNLI 类别）
id2label = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

def tokenize(sentence):
    """
    简单分词函数：英文按空格分词，中文按字分词
    """
    if all(ord(c) < 128 for c in sentence):
        return sentence.lower().split()  # 英文按空格
    else:
        return list(sentence.strip())    # 中文按字

def encode_sentence(sentence, word2idx, max_len):
    tokens = tokenize(sentence)
    ids = [word2idx.get(token, word2idx.get("<UNK>", 0)) for token in tokens]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def load_model(word2idx):
    model = ESIM(
        vocab_size=len(word2idx),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes
    )
    model.load_state_dict(torch.load(config.save_model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    return model

def predict(premise, hypothesis, model, word2idx):
    p_ids = encode_sentence(premise, word2idx, config.max_seq_len).unsqueeze(0).to(config.device)
    h_ids = encode_sentence(hypothesis, word2idx, config.max_seq_len).unsqueeze(0).to(config.device)

    with torch.no_grad():
        logits = model(p_ids, h_ids)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        return id2label[pred], confidence

if __name__ == "__main__":
    set_seed(config.random_seed)

    # 重建词表（必须和训练保持一致）
    print("📚 正在加载词表...")
    train_data = load_snli_data(config.train_file, max_samples=80000)
    word2idx = build_vocab(train_data, max_vocab_size=config.max_vocab_size)

    # 加载训练好的模型
    print("📦 正在加载模型参数...")
    model = load_model(word2idx)

    print("✨ 输入前提和假设句子进行预测（输入 q 退出）")
    while True:
        premise = input("\n请输入前提句子（Premise）：")
        if premise.strip().lower() in {"q", "quit", "exit"}:
            break
        hypothesis = input("请输入假设句子（Hypothesis）：")
        if hypothesis.strip().lower() in {"q", "quit", "exit"}:
            break

        label, conf = predict(premise, hypothesis, model, word2idx)
        print(f"🔎 预测结果：{label}（置信度：{conf:.4f}）")

