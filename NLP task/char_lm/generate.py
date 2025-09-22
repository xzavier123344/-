import torch
from model import CharRNN
from dataset import load_poetry_dataset
import config
import random

def generate(model, dataset, start_text='春', length=100, temperature=1.0, greedy=False):
    model.eval()
    device = torch.device(config.device)

    input_text = start_text
    input_ids = [dataset.char2idx.get(ch, 0) for ch in input_text]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    hidden = model.init_hidden(batch_size=1, device=device)

    # 先让模型读完初始字符
    for i in range(len(input_ids) - 1):
        _, hidden = model(input_tensor[:, i:i+1], hidden)

    # 用最后一个字符开始生成
    last_input = input_tensor[:, -1:]

    generated = input_text

    for _ in range(length):
        logits, hidden = model(last_input, hidden)  # (1, 1, vocab_size)
        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1).squeeze()

        if greedy:
            next_id = torch.argmax(probs).item()
        else:
            next_id = torch.multinomial(probs, num_samples=1).item()

        next_char = dataset.idx2char[next_id]
        generated += next_char
        last_input = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return generated

def main():
    # 加载数据集（用于词表）
    dataset = load_poetry_dataset(config.data_path, config.seq_len)

    # 加载模型
    model = CharRNN(
        vocab_size=dataset.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        rnn_type=config.rnn_type,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))

    # 生成文本
    for i in range(3):
        start = random.choice("春夏秋冬夜山月风花雨")
        poem = generate(model, dataset, start_text=start, length=100, temperature=0.8)
        print(f"\n📝 生成诗（起始：{start}）:\n{poem}\n")

if __name__ == '__main__':
    main()
