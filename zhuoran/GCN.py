# from torch_geometric.datasets import KarateClub
# import torch
# from torch.nn import Linear
# from torch_geometric.nn import GCNConv
#
# dataset = KarateClub()
# print(dataset[0])
#
# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         torch.manual_seed(1234)
#         self.conv1 = GCNConv(dataset.num_features, 4)
#         self.conv2 = GCNConv(4,4)
#         self.conv3 = GCNConv(4,2)
#         self.classifier = Linear(2, dataset.num_classes)
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = h.tanh()
#         h = self.conv2(h, edge_index)
#         h = h.tanh()
#         h = self.conv3(h, edge_index)
#         h = h.tanh()
#         out = self.classifier(h)
#         return out, h
#
# model = GCN()
#

from torch_geometric.datasets import KarateClub
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.optim as optim

# Load the dataset
dataset = KarateClub()
data = dataset[0]

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h

# Initialize the model
model = GCN()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train(data):
    model.train()
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
for epoch in range(1000):
    loss = train(data)
    print(f'Epoch: {epoch}, Loss: {loss:.4f}')

# Evaluation (optional)
model.eval()
_, h = model(data.x, data.edge_index)
pred = h.argmax(dim=1)
correct = pred.eq(data.y).sum().item()
accuracy = correct / data.num_nodes
print(f'Accuracy: {accuracy:.4f}')

