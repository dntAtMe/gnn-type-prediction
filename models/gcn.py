import torch
from torch.nn.functional import relu, dropout, softmax
from torch_geometric.nn import GCNConv, Linear


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, p, deep_layers=3):
        super().__init__()
        self.p = p

        self.conv_layers = torch.nn.ModuleList()

        for i in range(deep_layers):
            if i == 0:
                self.conv_layers.append(GCNConv(input_channels, hidden_channels))
            else:
                self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.fc = Linear(hidden_channels, output_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = relu(x)

        x = dropout(x, p=self.p, training=self.training)
        return self.fc(x)
