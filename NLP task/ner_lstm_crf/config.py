# 数据路径
TRAINING_DATA_PATH = "./data/resume_ner/train.txt"
VALIDATION_DATA_PATH = "./data/resume_ner/dev.txt"
TEST_DATA_PATH = "./data/resume_ner/test.txt"

# 模型保存路径
VOCAB_PATH = "./data/resume_ner/vocab.pkl"
MODEL_SAVE_PATH = "./check/bilstm_crf.pt"

# 模型超参数
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

# 特殊标记
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
