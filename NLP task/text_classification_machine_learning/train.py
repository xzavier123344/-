import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import load_data, build_vocab, vectorize
from model import LogisticRegression, SoftmaxRegression
import config

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def get_batches(X, y, batch_size, shuffle=True):
    m = X.shape[0]
    indices = np.arange(m)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, m, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def train(model, X_train, y_train, X_val, y_val, epochs, lr, batch_size):
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_batch, y_pred)
            dW, db = model.backward(X_batch, y_batch, y_pred)
            model.update(dW, db, lr)
            total_loss += loss

        avg_loss = total_loss / (len(X_train) // batch_size)
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Acc = {acc:.4f}")
    return model  # 返回最后模型

def save_model(model, path):
    np.savez(path, W=model.W, b=model.b)
    print(f"✅ 模型已保存到: {path}")

def main():
    # 固定随机种子
    np.random.seed(config.random_seed)

    # 加载数据
    texts, labels = load_data(config.train_file)
    vocab = build_vocab(texts, ngram=config.ngram, max_features=config.max_features)
    X = vectorize(texts, vocab, ngram=config.ngram)
    y = np.array(labels)

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.random_seed
    )

    # 初始化模型
    input_dim = X.shape[1]
    if config.model_type == 'logistic':
        y_train = (y_train > 2).astype(int)
        y_val = (y_val > 2).astype(int)
        model = LogisticRegression(input_dim)
    else:
        num_classes = len(set(y))
        model = SoftmaxRegression(input_dim, num_classes)

    # 训练并保存模型
    model = train(
        model, X_train, y_train, X_val, y_val,
        epochs=config.num_epochs,
        lr=config.learning_rate,
        batch_size=config.batch_size
    )

    save_model(model, config.save_model_path)

if __name__ == '__main__':
    main()

