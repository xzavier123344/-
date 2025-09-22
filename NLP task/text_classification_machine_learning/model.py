import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 防止溢出
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """
    y_true: shape (batch_size,)
    y_pred: shape (batch_size, num_classes)
    """
    m = y_true.shape[0]
    log_probs = -np.log(y_pred[range(m), y_true] + 1e-15)
    return np.sum(log_probs) / m

class LogisticRegression:
    def __init__(self, input_dim):
        self.W = np.zeros((input_dim, 1))
        self.b = 0

    def forward(self, X):
        z = X @ self.W + self.b
        return sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = - (y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return np.mean(loss)

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        dz = y_pred - y_true.reshape(-1, 1)
        dW = X.T @ dz / m
        db = np.sum(dz) / m
        return dW, db

    def update(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db

    def predict(self, X):
        prob = self.forward(X)
        return (prob >= 0.5).astype(int)

class SoftmaxRegression:
    def __init__(self, input_dim, num_classes):
        self.W = np.zeros((input_dim, num_classes))
        self.b = np.zeros((1, num_classes))

    def forward(self, X):
        z = X @ self.W + self.b
        return softmax(z)

    def compute_loss(self, y_true, y_pred):
        return cross_entropy_loss(y_true, y_pred)

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[range(m), y_true] = 1
        dz = (y_pred - y_one_hot) / m
        dW = X.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        return dW, db

    def update(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
