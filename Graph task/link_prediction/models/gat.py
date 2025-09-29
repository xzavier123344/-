import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)

    def encode(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)