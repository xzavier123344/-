# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_random_embedding

class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(ESIM, self).__init__()
        self.embedding = init_random_embedding(vocab_size, embedding_dim)

        self.encode_lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)

        self.infer_lstm = nn.LSTM(hidden_size * 8, hidden_size, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, p, h):
        # 1. Embedding
        p_embed = self.embedding(p)  # [B, L, D]
        h_embed = self.embedding(h)

        # 2. Encode
        p_out, _ = self.encode_lstm(p_embed)  # [B, L, 2H]
        h_out, _ = self.encode_lstm(h_embed)

        # 3. Attention (token-to-token)
        attention = torch.matmul(p_out, h_out.transpose(1, 2))  # [B, Lp, Lh]

        p_align = torch.matmul(F.softmax(attention, dim=-1), h_out)      # [B, Lp, 2H]
        h_align = torch.matmul(F.softmax(attention.transpose(1, 2), dim=-1), p_out)  # [B, Lh, 2H]

        # 4. Enhancement (compare)
        def enhance(a, b):
            return torch.cat([
                a,
                b,
                a - b,
                a * b
            ], dim=-1)

        p_combined = enhance(p_out, p_align)  # [B, L, 8H]
        h_combined = enhance(h_out, h_align)

        # 5. Inference composition
        p_inf, _ = self.infer_lstm(p_combined)
        h_inf, _ = self.infer_lstm(h_combined)

        # 6. Pooling
        def pool(x):
            return torch.cat([
                torch.max(x, dim=1).values,
                torch.mean(x, dim=1)
            ], dim=-1)  # [B, 4H]

        v = pool(p_inf)
        u = pool(h_inf)
        final = torch.cat([v, u], dim=-1)  # [B, 8H]

        # 7. Classify
        logits = self.classifier(self.dropout(final))  # [B, C]
        return logits
