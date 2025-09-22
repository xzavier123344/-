import torch
import config
import pickle
from model import BiLSTM_CRF


def load_model():
    # 加载字典
    with open(config.VOCAB_PATH, 'rb') as f:
        word2id, tag2id, id2tag = pickle.load(f)

    # 模型结构
    model = BiLSTM_CRF(
        vocab_size=len(word2id),
        tagset_size=len(tag2id),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM
    )

    # 加载权重
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location='cpu'))
    model.eval()
    return model, word2id, id2tag


def predict(sentence, model, word2id, id2tag):
    tokens = list(sentence)
    input_ids = [word2id.get(char, word2id[config.UNK_TOKEN]) for char in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)  # shape (1, seq_len)
    mask = torch.ones_like(input_tensor, dtype=torch.bool)

    with torch.no_grad():
        pred = model(input_tensor, mask=mask)[0]  # batch size 1

    tags = [id2tag[i] for i in pred]
    return list(zip(tokens, tags))


if __name__ == "__main__":
    model, word2id, id2tag = load_model()

    print("👉 输入句子进行实体识别（输入 q 退出）")
    while True:
        sentence = input("\n请输入句子：")
        if sentence.lower() in {'q', 'quit', 'exit'}:
            break
        results = predict(sentence, model, word2id, id2tag)
        print("\n🔍 实体识别结果：")
        for char, tag in results:
            print(f"{char}  -->  {tag}")
