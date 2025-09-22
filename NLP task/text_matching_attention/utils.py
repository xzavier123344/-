import numpy as np
import torch
import torch.nn as nn
import random

def set_seed(seed=42):
    """
    设置随机种子，确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_random_embedding(vocab_size, embedding_dim):
    """
    创建随机初始化的词向量矩阵，用于 nn.Embedding
    """
    embedding = nn.Embedding(vocab_size, embedding_dim)
    nn.init.uniform_(embedding.weight, a=-0.05, b=0.05)  # 可根据任务调整范围
    return embedding

def accuracy(y_true, y_pred):
    """
    计算准确率
    """
    return (y_true == y_pred).sum().item() / len(y_true)