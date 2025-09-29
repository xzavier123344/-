import torch
import torch.nn.functional as F
from evaluate import evaluate
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling

def train_sampler(model, data, batch_size=1024, epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    data = data.to(device)

    # 定义采样器：采样节点的邻居子图
    loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        input_nodes=None,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)

            # 对整个 batch 构建子图嵌入
            z = model.encode(batch.x, batch.edge_index)

            # 正边
            pos_edge_index = data.train_pos_edge_index
            pos_pred = model.decode(z, pos_edge_index)
            pos_label = torch.ones(pos_pred.size(0), device=device)

            # 负边
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_pred.size(0)
            )
            neg_pred = model.decode(z, neg_edge_index)
            neg_label = torch.zeros(neg_pred.size(0), device=device)

            pred = torch.cat([pos_pred, neg_pred])
            label = torch.cat([pos_label, neg_label])

            loss = F.binary_cross_entropy_with_logits(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        auc, ap = evaluate(model, data)
        print(f"[Sampler] Epoch {epoch:03d}, Loss: {total_loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
