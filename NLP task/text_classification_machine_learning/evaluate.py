import numpy as np
from preprocess import load_data, build_vocab, vectorize
from model import LogisticRegression, SoftmaxRegression
from train import accuracy_score
from config import *

def load_model(model_type, input_dim, num_classes, model_path):
    if model_type == 'logistic':
        model = LogisticRegression(input_dim)
    else:
        model = SoftmaxRegression(input_dim, num_classes)
    data = np.load(model_path)
    model.W = data['W']
    model.b = data['b']
    return model

def main():
    # 加载测试数据
    texts, labels = load_data(train_file)
    vocab = build_vocab(texts, ngram=ngram, max_features=max_features)
    X = vectorize(texts, vocab, ngram=ngram)
    y = np.array(labels)

    # 模型准备
    input_dim = X.shape[1]
    if model_type == 'logistic':
        y = (y > 2).astype(int)
        model = load_model('logistic', input_dim, 2, save_model_path)
    else:
        num_classes = len(set(y))
        model = load_model('softmax', input_dim, num_classes, save_model_path)

    # 预测与评估
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f" 模型在完整训练集上的准确率: {acc:.4f}")

    # ：展示前几条预测
    for i in range(5):
        print(f"句子: {texts[i]}")
        print(f"预测: {y_pred[i]}, 实际: {y[i]}")
        print("---")

if __name__ == "__main__":
    main()
