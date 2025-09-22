import torch

# 数据路径
train_file = 'data/snli_1.0/snli_1.0_train.jsonl'
dev_file   = 'data/snli_1.0/snli_1.0_dev.jsonl'
test_file  = 'data/snli_1.0/snli_1.0_test.jsonl'

# 词表 & embedding
embedding_dim = 100
max_vocab_size = 20000
max_seq_len = 50
use_glove = False             # 若使用 GloVe 设为 True

# 模型结构
model_type = 'esim'           # 可拓展其他模型
hidden_size = 128
num_classes = 3               # entailment / neutral / contradiction

# 训练超参数
batch_size = 64
num_epochs = 10
learning_rate = 1e-3
dropout = 0.5

# 保存模型路径
save_model_path = 'saved_models/esim_snli.pt'

# 设备设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 随机种子
random_seed = 42