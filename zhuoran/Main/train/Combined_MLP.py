import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

torch.set_printoptions(precision=8)

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels_2, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels_2)
        self.conv3 = GCNConv(hidden_channels_2, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        h = F.relu(x)
        x = self.conv3(h, edge_index)
        return x, h

# Prepare the data
def prepare_data(features, adj_matrix, label):
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(label, dtype=torch.float)

    edge_index = []
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] != 0:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)


model_GCN = GCN(in_channels=2, hidden_channels=4, hidden_channels_2=4, out_channels=2)
model_GCN.load_state_dict(torch.load('GCN_best_model_2out_1024.pth'))
model_GCN.eval()