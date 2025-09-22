import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from preprocess import build_vocab, TextDataset, load_data
from utils import load_glove_embeddings
from model import TextCNN, BiLSTM
import config
import os
from sklearn.model_selection import train_test_split


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)


def train():
    set_seed(config.random_seed)
    device = torch.device(config.device)

    # 加载数据
    texts, labels = load_data(config.train_file)
    word2idx = build_vocab(texts, max_vocab_size=config.max_vocab_size)

    # Embedding
    if config.use_glove:
        embedding_matrix = load_glove_embeddings(config.glove_path, word2idx, config.embedding_dim)
    else:
        embedding_matrix = None

    # 构建 dataset
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=config.random_seed)
    train_dataset = TextDataset(X_train, y_train, word2idx, config.max_len)
    val_dataset = TextDataset(X_val, y_val, word2idx, config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 模型选择
    if config.model_type == 'cnn':
        model = TextCNN(len(word2idx), config.embedding_dim, config.num_classes,
                        embedding_matrix=embedding_matrix, dropout=config.dropout)
    else:
        model = BiLSTM(len(word2idx), config.embedding_dim, config.hidden_dim, config.num_classes,
                       embedding_matrix=embedding_matrix, dropout=config.dropout)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_acc = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                logits = model(val_x)
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(val_y)

        val_acc = accuracy(torch.tensor(all_labels), torch.tensor(all_preds))
        print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(config.save_model_path), exist_ok=True)
            torch.save(model.state_dict(), config.save_model_path)
            #print(f"✅ 新最佳模型已保存 (Val Acc: {val_acc:.4f})")


if __name__ == '__main__':
    train()