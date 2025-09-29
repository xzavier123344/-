import torch
import torch.nn.functional as F
from evaluate import evaluate
from torch_geometric.utils import negative_sampling

def train_full(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        z = model.encode(data.x, data.train_pos_edge_index)

        # 正边预测
        pos_pred = model.decode(z, data.train_pos_edge_index)
        pos_label = torch.ones(pos_pred.size(0), device=pos_pred.device)

        # 负边采样
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_pred.size(0)
        )
        neg_pred = model.decode(z, neg_edge_index)
        neg_label = torch.zeros(neg_pred.size(0), device=neg_pred.device)

        # 拼接损失
        pred = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(pred, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0:
            auc, ap = evaluate(model, data)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
