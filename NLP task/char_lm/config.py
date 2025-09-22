# 数据路径
data_path = './poetryFromTang.txt'

# 模型保存路径
save_path = 'char_rnn_model.pt'

# 模型类型：'lstm' or 'gru'
rnn_type = 'lstm'

# 模型结构
embedding_dim = 128
hidden_dim = 256
num_layers = 2
dropout = 0.2
seq_len = 100

# 训练超参数
batch_size = 64
num_epochs = 20
learning_rate = 0.002
clip_grad = 5.0

# 设备
device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
