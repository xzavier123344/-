import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, ReLU
import torch

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GIN, self).__init__()
        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)