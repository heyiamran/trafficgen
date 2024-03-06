import pickle
from enum import Enum
import numpy as np

class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    @staticmethod
    def is_road_line(line):
        return True if line.__class__ == RoadLineType else False

    @staticmethod
    def is_yellow(line):
        return True if line in [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        return True if line in [
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_SINGLE_WHITE
        ] else False


class RoadEdgeType(Enum):
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    @staticmethod
    def is_road_edge(edge):
        return True if edge.__class__ == RoadEdgeType else False

    @staticmethod
    def is_sidewalk(edge):
        return True if edge == RoadEdgeType.BOUNDARY else False


class AgentType(Enum):
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4






def distance(pos1, pos2):
    # Calculate Euclidean distance between two positions
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def construct_heterogeneous_graph(positions, threshold):
    num_agents = len(positions)

    # Initialize node set and adjacency matrix
    V = list(range(num_agents))
    A = np.zeros((num_agents, num_agents))

    # Iterate over all pairs of agents
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if distance(positions[i], positions[j]) < threshold:
                A[i][j] = 1
                A[j][i] = 1

    return V, A

if __name__ == "__main__":
    file_path = '0.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        # print(type(data))
        # print(data.keys())
        # print("id: ",data['id'])
        # print("all_agent: ", data['all_agent'][0][0])
        #
        # print("traffic_light situation: ", len(data['traffic_light']))
        # print("traffic_light: ", data['traffic_light'])
        #
        # print("lane number: ", len(data['lane']))
        # print("lane: ", data['lane'][0])
        # print("lane: ", data['lane'][1])
        # print("lane: ", data['lane'][1200])

        # Use the data that the time point is at 0.
        # i = data['all_agent'].shape[1]
        # print("all_agent: ", data['all_agent'][0][i-1])
        # print("traffic_light: ", data['traffic_light'][0])
        # print("lane:", data['lane'][0])

        #lanes need to be modified, cut into pieces if the length is great than some limit.
        # print("center_info:", type(data['center_info']))
        # for key in data['center_info'].keys():
        #     print(key)
        #
        # print("center_info:", data['center_info'][446]['width'])

        # LaneType
        # {
        # TYPE_UNDEFINED = 0;
        # TYPE_FREEWAY = 1;
        # TYPE_SURFACE_STREET = 2;
        # TYPE_BIKE_LANE = 3;
        # }
        # line center position (x,y),
        SURFACE_STREET = []
        for item in data['lane']:
            if item[2] == 2:
                print(item)
                SURFACE_STREET.append(item)


# This code below is how to construct the graph with only agents(no lanes)
    # Example input data, Use the data that the time point is at 0.
    #input_data = data['all_agent'][0] + SURFACE_STREET

    combined_list = []

    for sublist in data['all_agent'][0]:
        combined_list.append(sublist[:2])

    for sublist in SURFACE_STREET:
        combined_list.append(sublist[:2])

    input_data = combined_list

    # Customized threshold
    threshold = 10.0  # Customize threshold as needed

    # Convert input data to numpy array
    positions = np.array(input_data)

    # Construct the heterogeneous graph
    V, A = construct_heterogeneous_graph(positions, threshold)

    print("Node set V:", V)
    print("Adjacency matrix A:")
    print(A)

