import torch
import torch_geometric
from torch.nn.functional import relu, dropout


class GGNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers, p):
        super(GGNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.p = p

        self.lin = torch_geometric.nn.Linear(input_channels, hidden_channels)
        self.gate1 = torch_geometric.nn.GatedGraphConv(hidden_channels, num_layers)
        self.lin2 = torch_geometric.nn.Linear(hidden_channels, output_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin(x)
        x = self.gate1(x, edge_index)

        x = relu(x)
        x = dropout(x, p=self.p, training=self.training)
        x = self.lin2(x)

        return x