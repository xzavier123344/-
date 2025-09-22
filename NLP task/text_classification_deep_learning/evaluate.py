import torch
from torch.utils.data import DataLoader
from preprocess import load_data, build_vocab, TextDataset
from utils import load_glove_embeddings
from model import TextCNN, BiLSTM
import config

def accuracy(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)

def load_model(vocab_size, embedding_matrix):
    if config.model_type == 'cnn':
        model = TextCNN(vocab_size, config.embedding_dim, config.num_classes,
                        embedding_matrix=embedding_matrix, dropout=config.dropout)
    else:
        model = BiLSTM(vocab_size, config.embedding_dim, config.hidden_dim,
                       config.num_classes, embedding_matrix=embedding_matrix, dropout=config.dropout)

    model.load_state_dict(torch.load(config.save_model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    print("✅ 已加载模型:", config.save_model_path)
    return model

def main():
    # 加载数据（使用训练集中的一部分来验证）
    texts, labels = load_data(config.train_file)
    word2idx = build_vocab(texts, max_vocab_size=config.max_vocab_size)

    # GloVe or random
    if config.use_glove:
        embedding_matrix = load_glove_embeddings(config.glove_path, word2idx, config.embedding_dim)
    else:
        embedding_matrix = None

    # 验证集构造（选用最后20%）
    val_texts = texts[-3000:]
    val_labels = labels[-3000:]
    val_dataset = TextDataset(val_texts, val_labels, word2idx, config.max_len)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 加载模型
    model = load_model(len(word2idx), embedding_matrix)

    # 验证评估
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(config.device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.tolist())

    acc = accuracy(torch.tensor(all_labels), torch.tensor(all_preds))
    print(f"🎯 模型在验证集上的准确率: {acc:.4f}")

    # 示例输出
    for i in range(5):
        print(f"文本: {val_texts[i]}")
        print(f"预测: {all_preds[i]}, 实际: {val_labels[i]}")
        print("---")

if __name__ == '__main__':
    main()
