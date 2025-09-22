# model.py

import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, rnn_type='lstm', num_layers=2, dropout=0.2):
        super(CharRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout)
        else:
            raise ValueError("Unsupported rnn_type: choose 'lstm' or 'gru'.")

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len)
        hidden: initial hidden state
        """
        embed = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(embed, hidden)  # output: (batch, seq_len, hidden_dim)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
