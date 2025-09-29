import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class GNNClassifier(torch.nn.Module):
    def __init__(self, model_type, pooling, input_dim, output_dim, task_type='classification', hidden_dim=64):
        super().__init__()
        self.pooling = pooling
        self.task_type = task_type

        if model_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif model_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=1)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)
        elif model_type == 'GraphSAGE':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        elif model_type == 'GIN':
            nn1 = Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        else:
            raise ValueError("Unsupported model type.")

        self.lin = Linear(hidden_dim, output_dim)

    def pool(self, x, batch):
        if self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        elif self.pooling == 'min':
            return -global_max_pool(-x, batch)
        else:
            raise ValueError("Unsupported pooling type.")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()  # 确保特征是 float 类型
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        out = self.lin(x)
        return out.squeeze() if self.task_type == 'regression' else out