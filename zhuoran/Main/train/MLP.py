import pickle

with open('../tools/agent_feature_map.pkl', 'rb') as file:
    agent_feature_map = pickle.load(file)

with open('../tools/agent_label.pkl', 'rb') as file:
    agent_label = pickle.load(file)


agent_feature_map_normalized = agent_feature_map
agent_label_normalized = agent_label

print(agent_feature_map_normalized[0])
print(agent_label_normalized[0])