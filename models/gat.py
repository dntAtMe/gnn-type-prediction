from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch


class GAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, heads, p=0.6):
        super().__init__()
        self.p = p

        self.conv1 = GATConv(input_channels, hidden_channels, heads=heads, dropout=self.p)
        self.conv2 = GATConv(hidden_channels * heads, output_channels, heads=1, dropout=self.p)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)