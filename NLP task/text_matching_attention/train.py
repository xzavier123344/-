import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import config
from utils import set_seed, accuracy
from dataset import load_snli_data, build_vocab, SNLIDataset
from model import ESIM

def train():
    set_seed(config.random_seed)
    device = torch.device(config.device)

    # 1. 加载数据
    print("🔍 加载 SNLI 数据...")
    train_data = load_snli_data(config.train_file, max_samples=80000)
    dev_data = load_snli_data(config.dev_file, max_samples=10000)

    # 2. 构建词表
    word2idx = build_vocab(train_data, max_vocab_size=config.max_vocab_size)

    # 3. 构造数据集和加载器
    train_set = SNLIDataset(train_data, word2idx, config.max_seq_len)
    dev_set = SNLIDataset(dev_data, word2idx, config.max_seq_len)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=config.batch_size)

    # 4. 构建模型
    model = ESIM(
        vocab_size=len(word2idx),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 5. 训练过程
    best_dev_acc = 0.0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0

        for p_batch, h_batch, y_batch in train_loader:
            p_batch = p_batch.to(device)
            h_batch = h_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(p_batch, h_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 6. 验证
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for p_batch, h_batch, y_batch in dev_loader:
                p_batch = p_batch.to(device)
                h_batch = h_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(p_batch, h_batch)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y_batch.cpu().tolist())

        dev_acc = accuracy(torch.tensor(all_labels), torch.tensor(all_preds))
        print(f"📘 Epoch {epoch} | Loss: {total_loss:.4f} | Dev Acc: {dev_acc:.4f}")

        # 7. 保存最优模型
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), config.save_model_path)
            print(f"✅ 模型已保存，当前最佳验证准确率: {dev_acc:.4f}")

if __name__ == '__main__':
    train()
