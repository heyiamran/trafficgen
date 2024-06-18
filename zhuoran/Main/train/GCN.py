import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Load data
with open('../tools/agent_feature_graph.pkl', 'rb') as file:
    agent_feature_graph = pickle.load(file)

with open('../tools/agent_label.pkl', 'rb') as file:
    agent_label = pickle.load(file)

agent_feature_graph_normalized = agent_feature_graph
agent_label_normalized = agent_label

adj_matrixs = []
GCN_feature = []
GCN_label = []

# Function to calculate Euclidean distance
def euclidean_distance(coord1, coord2):
    return np.sqrt(np.sum((coord1 - coord2)**2))

# Function to construct adjacency matrix
def construct_adjacency_matrix(ego_coord, other_coords):
    num_cars = len(other_coords)
    adjacency_matrix = np.zeros((6, 6))  # Initialize adjacency matrix with shape (6, 6)

    # Calculate distances between ego car and other cars
    distances = np.array([euclidean_distance(ego_coord, coord) for coord in other_coords])

    # Get indices of closest 5 cars
    closest_indices = np.argsort(distances)[:5]

    # Update adjacency matrix with connections to closest 5 cars
    for i, index in enumerate(closest_indices):
        adjacency_matrix[0, i+1] = distances[index]  # Distance from ego car to other car
        adjacency_matrix[i+1, 0] = distances[index]  # Distance from other car to ego car

    return closest_indices, adjacency_matrix

# Loop through data to prepare for GCN
for i in range(len(agent_feature_graph_normalized)):
    ego = agent_feature_graph_normalized[i][0][:5]
    other_cars = [car[:2] for car in agent_feature_graph_normalized[i][1:][:5]]

    ego_car = np.array(ego[:2])  # Ego car coordinates

    # Construct the adjacency matrix
    closest_indices, adj_matrix = construct_adjacency_matrix(ego_car, other_cars)
    adj_matrixs.append(adj_matrix)

    node_embed = []
    node_label = []

    node_embed.append(agent_feature_graph_normalized[i][0][:5])
    node_label.append(agent_label_normalized[i][:5])
    for j in closest_indices:
        node_embed.append(agent_feature_graph_normalized[i][j][:5])

    GCN_feature.append(node_embed)
    GCN_label.append(node_label)

# Convert to PyTorch tensors
GCN_feature = torch.tensor(GCN_feature, dtype=torch.float)
GCN_label = torch.tensor(GCN_label, dtype=torch.float)

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

# Move data and model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

GCN_feature = GCN_feature.to(device)
GCN_label = GCN_label.to(device)

adj_matrixs = [torch.tensor(adj_matrix, dtype=torch.float).to(device) for adj_matrix in adj_matrixs]

# Prepare data loader
data_list = [Data(x=GCN_feature[i], edge_index=adj_matrixs[i].nonzero(as_tuple=False).t().contiguous(), y=GCN_label[i]) for i in range(len(GCN_feature))]
loader = DataLoader(data_list, batch_size=1)

# Initialize the model, optimizer, and loss function
model = GCN(in_channels=2, hidden_channels=4, hidden_channels_2=4, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

best_loss = float('inf')
best_model = None
losses = []

# Training loop
model.train()

epoch_num = 1000
for epoch in range(epoch_num):
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, h = model(data.x, data.edge_index)
        loss = loss_fn(out[0].unsqueeze(0), data.y)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model.state_dict()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Save the best model
torch.save(best_model, 'GCN_best_model_2out.pth')

# Load the best model for testing
model.load_state_dict(torch.load('GCN_best_model_2out_1024.pth'))

# Testing the model
model.eval()
for data in loader:
    data = data.to(device)
    out, h = model(data.x, data.edge_index)
    print(f'Predicted: {out.detach().cpu().numpy()}, Actual: {data.y.cpu().numpy()}')

# Plotting the loss
plt.plot(range(epoch_num), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.savefig('training_loss_per_epoch_GCN.png')
plt.show()

print("Minimum loss", min(losses))
