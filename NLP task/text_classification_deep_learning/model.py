import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes,
                 embedding_matrix=None, kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.unsqueeze(1)     # [batch, 1, seq_len, embed_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # list of [batch, num_filters, seq_len-k+1]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]   # list of [batch, num_filters]
        x = torch.cat(x, 1)     # [batch, num_filters * len(kernel_sizes)]
        x = self.dropout(x)
        return self.fc(x)       # [batch, num_classes]


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 embedding_matrix=None, num_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        output, _ = self.lstm(embedded)  # output: [batch, seq_len, hidden*2]
        out = self.dropout(output[:, -1, :])  # 使用最后时间步的输出
        return self.fc(out)  # [batch, num_classes]
