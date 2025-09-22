train_file = 'data/train.tsv'
test_file = 'data/test.tsv'

# 模型配置
model_type = 'softmax'   # 'logistic' or 'softmax'
ngram = 1
max_features = 5000

# 训练参数
learning_rate = 0.1
batch_size = 64
num_epochs = 10
random_seed = 42

# 模型保存路径
save_model_path = 'saved_model.npz'
