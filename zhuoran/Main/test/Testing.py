import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

with open('../tools/agent_feature_map_test.pkl', 'rb') as file:
    agent_feature_map_test = pickle.load(file)

with open('../tools/agent_label_test.pkl', 'rb') as file:
    agent_label_test = pickle.load(file)

with open('../tools/agent_feature_graph_test.pkl', 'rb') as file:
    agent_feature_graph_test = pickle.load(file)

adj_matrixs_test= []
GCN_feature_test = []
GCN_label_test = []
for i in range(len(agent_feature_graph_test)):
  ego = agent_feature_graph_test[i][0][:5]
  other_cars = list()
  for car in agent_feature_graph_test[i][1:][:5]:
    other_cars.append(car[:2])
  import numpy as np

  def euclidean_distance(coord1, coord2):
      return np.sqrt(np.sum((coord1 - coord2)**2))

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

      return closest_indices,adjacency_matrix


  ego_car = np.array(ego[:2])  # Ego car coordinates
  other_cars = other_cars  # Coordinates of other cars

  # Construct the adjacency matrix
  closest_indices, adj_matrix = construct_adjacency_matrix(ego_car, other_cars)

  closest_indices = closest_indices+1
  adj_matrixs_test.append(adj_matrix)

  node_embed = []
  node_label = []

  node_embed.append(agent_feature_graph_test[i][0][:5])
  node_label.append(agent_label_test[i][:5])
  for j in closest_indices:
    node_embed.append(agent_feature_graph_test[i][j][:5])
    #node_label.append(data['all_agent'][time_point][i])

  GCN_feature_test.append(node_embed)
  GCN_label_test.append(node_label)


import torch
import torch.nn as nn


# Define the MLP model without batch normalization
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_prob=0.5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        hidden_output = self.relu3(x)
        x = self.dropout(hidden_output)
        x = self.fc4(x)
        return x, hidden_output

# Define input and label data
agent_feature_map_normalized = torch.tensor(agent_feature_map_test, dtype=torch.float)
agent_label_normalized = torch.tensor(agent_label_test, dtype=torch.float)

# Define model parameters
input_size = agent_feature_map_normalized.shape[1]
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
output_size = agent_label_normalized.shape[1]

# Instantiate the model
model_cMLP = SimpleMLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Load the trained model
model_path = '../train/simple_mlp_2out_1024.pt'
model_cMLP.load_state_dict(torch.load(model_path))
model_cMLP.eval()  # Set the model to evaluation mode





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
model_GCN.load_state_dict(torch.load('../train/GCN_best_model_2out_1024.pth'))
model_GCN.eval()


data_list = [prepare_data(GCN_feature_test[i], adj_matrixs_test[i], GCN_label_test[i]) for i in range(len(GCN_feature_test))]
loader = DataLoader(data_list, batch_size=1)

hidden_states_test = []
for data in loader:
  output, hidden_state = model_GCN(data.x, data.edge_index)
  hidden_states_test.append(hidden_state[0].tolist())

agent_feature_combined = [sublist1 + sublist2 for sublist1, sublist2 in zip(agent_feature_map_test, hidden_states_test)]


agent_feature_map_normalized_t = agent_feature_combined
agent_label_normalized_t = agent_label_test


def position_diff(list1, list2):
  return np.abs(list1[0] - list2[0]) + np.abs(list1[1] - list2[1])

total_diff = 0
for test_id in range(10):
  with torch.no_grad():
    output, hidden_output = model_cMLP(torch.tensor(agent_feature_map_normalized_t[test_id], dtype=torch.float))
    predictions = output.numpy()

  total_diff += position_diff(output.tolist(), agent_label_normalized_t[test_id])

average_diff = total_diff/10
print(average_diff)





import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

# Define the MLP model without batch normalization
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_prob=0.5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        hidden_output = self.relu3(x)
        x = self.dropout(hidden_output)
        x = self.fc4(x)
        return x, hidden_output

# Define input and label data
agent_feature_map_normalized = torch.tensor(agent_feature_map_test, dtype=torch.float)
agent_label_normalized = torch.tensor(agent_label_test, dtype=torch.float)

# Define model parameters
input_size = agent_feature_map_normalized.shape[1]
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
output_size = agent_label_normalized.shape[1]

# Instantiate the model
model_MLP = SimpleMLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Load the trained model
model_path = '../train/simple_mlp_2out_best_1024.pt'
model_MLP.load_state_dict(torch.load(model_path))
model_MLP.eval()  # Set the model to evaluation mode


def position_diff(list1, list2):
  return np.abs(list1[0] - list2[0]) + np.abs(list1[1] - list2[1])

total_diff = 0
for test_id in range(10):
  with torch.no_grad():
    output, hidden_output = model_MLP(torch.tensor(agent_feature_map_test[test_id], dtype=torch.float))
    predictions = output.numpy()

  total_diff += position_diff(output.tolist(), agent_label_normalized_t[test_id])

average_diff = total_diff/10
print(average_diff)