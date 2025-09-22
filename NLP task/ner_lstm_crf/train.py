import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import NERDataset, collate_fn
from model import BiLSTM_CRF
import pickle
import config
import os

def evaluate(model, dataloader, id2tag):
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y, mask in dataloader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            preds = model(X, mask=mask)
            for i in range(len(X)):
                true = y[i][mask[i]].tolist()
                pred = preds[i]
                y_true.extend([id2tag[t] for t in true])
                y_pred.extend([id2tag[t] for t in pred])

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return precision, recall, f1

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载词典
    with open(config.VOCAB_PATH, 'rb') as f:
        word2id, tag2id, id2tag = pickle.load(f)

    # 加载数据集
    train_dataset = NERDataset(config.TRAINING_DATA_PATH, word2id, tag2id)
    val_dataset = NERDataset(config.VALIDATION_DATA_PATH, word2id, tag2id)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

    # 构建模型
    model = BiLSTM_CRF(len(word2id), len(tag2id),
                       embedding_dim=config.EMBEDDING_DIM,
                       hidden_dim=config.HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_f1 = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for X, y, mask in train_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            loss = model(X, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\nEpoch {epoch+1}/{config.EPOCHS} - Training Loss: {total_loss:.4f}")

        # 验证集评估
        precision, recall, f1 = evaluate(model, val_loader, id2tag)
        print(f"Validation → Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # 保存最优模型
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"✅ Best model saved to {config.MODEL_SAVE_PATH} with F1: {f1:.4f}\n")

    print("🎉 Training complete.")

if __name__ == "__main__":
    train()




