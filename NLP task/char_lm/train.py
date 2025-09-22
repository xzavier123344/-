# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import load_poetry_dataset
from model import CharRNN
import config
import math
import os

def train():
    # 设备
    device = torch.device(config.device)

    # 数据加载
    dataset = load_poetry_dataset(config.data_path, seq_len=config.seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    vocab_size = dataset.vocab_size
    print(f"📘 加载数据完成，共 {len(dataset)} 条样本，字符集大小：{vocab_size}")

    # 初始化模型
    model = CharRNN(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        rnn_type=config.rnn_type,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            hidden = model.init_hidden(batch_size=batch_x.size(0), device=device)
            logits, _ = model(batch_x, hidden)

            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss)

        print(f"📘 Epoch {epoch}/{config.num_epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

        # 保存模型
        torch.save(model.state_dict(), config.save_path)
        print(f"✅ 模型保存至：{config.save_path}")

if __name__ == '__main__':
    train()
