# 数据路径
train_file = 'data/train.tsv'
glove_path = 'data/glove.6B.100d.txt'
save_model_path = 'saved_models/best_model.pt'

# 模型类型: 'cnn' or 'bilstm'
model_type = 'cnn'

# 训练参数
embedding_dim = 100
hidden_dim = 128       # 用于 BiLSTM
num_classes = 5
num_epochs = 10
batch_size = 64
learning_rate = 5e-3
dropout = 0.5
max_len = 50
max_vocab_size = 5000

# 词嵌入
use_glove = True

# 设备
device = 'cuda'  # or 'cpu'
random_seed = 42