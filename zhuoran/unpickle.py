import pickle
import tensorflow as tf
from enum import Enum
import sys

from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    pass
import scenario_pb2
import os
import pickle
import numpy as np
from enum import Enum
import scenario_pb2

# Get from trafficgen/utils/typedef.py
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

#-----------------------------------------------------------

SAMPLE_NUM = 10
LANE_DIM = 4
TIME_SAMPLE = 3  # sample 64 time step, 64*3 = 192
BATCH_SIZE = 190  # 64*3 < 192, ok

def extract_tracks(f, sdc_index):
    # agents = np.zeros([len(f), BATCH_SIZE, 9])
    agents = np.zeros([len(f), BATCH_SIZE, 9])
    # sdc = f[sdc_index]
    # sdc_x = np.array([state.center_x for state in sdc.states])
    # sdc_y = np.array([state.center_y for state in sdc.states])
    # sdc_heading = np.array([state.heading for state in sdc.states])
    # sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)

    for i in range(len(f)):
        x = np.array([state.center_x for state in f[i].states])
        y = np.array([state.center_y for state in f[i].states])
        # pos = np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1)], axis=-1)
        l = np.array([[state.length] for state in f[i].states])[:, 0]
        w = np.array([[state.width] for state in f[i].states])[:, 0]
        head = np.array([state.heading for state in f[i].states])
        vx = np.array([state.velocity_x for state in f[i].states])
        vy = np.array([state.velocity_y for state in f[i].states])
        valid = np.array([[state.valid] for state in f[i].states])[:, 0]
        t = np.repeat(f[i].object_type, len(valid))
        agents[i] = np.stack((x, y, vx, vy, head, l, w, t, valid), axis=-1)[:BATCH_SIZE]
        # agents[i] = np.concatenate((pos, velocity, head, l, w, t, valid), axis=-1)[:BATCH_SIZE]

    ego = agents[[sdc_index]]
    others = np.delete(agents, sdc_index, axis=0)
    # others = np.transpose(others,[1,0,2])
    # others[abs(others[..., 0] > 80), -1] = 0
    # others[abs(others[..., 1] > 80), -1] = 0
    all_agent = np.concatenate([ego, others], axis=0)
    return all_agent.swapaxes(0, 1)


def yaw_to_theta(angles, thetas):
    """
    In time horizon
    """
    ret = []
    for theta, angle in zip(thetas, angles):
        theta = trans_angle(theta)
        angle -= theta
        angle = trans_angle(angle)
        if angle >= np.pi:
            angle -= 2 * np.pi
        ret.append(angle)
    return np.array(ret)

    # transform angles into 0~2pi


def trans_angle(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle


def transform_coord(coords, angle):
    x = coords[..., 0]
    y = coords[..., 1]
    x_transform = np.cos(angle) * x - np.sin(angle) * y
    y_transform = np.cos(angle) * y + np.sin(angle) * x
    output_coords = np.stack((x_transform, y_transform), axis=-1)

    if coords.shape[1] == 3:
        output_coords = np.concatenate((output_coords, coords[:, 2:]), axis=-1)
    return output_coords


def extract_tracks(f, sdc_index): # scenario.tracks, sdc_index The object states for a single object through the scenario
    # agents = np.zeros([len(f), BATCH_SIZE, 9])
    agents = np.zeros([len(f), BATCH_SIZE, 9])
    # sdc = f[sdc_index]
    # sdc_x = np.array([state.center_x for state in sdc.states])
    # sdc_y = np.array([state.center_y for state in sdc.states])
    # sdc_heading = np.array([state.heading for state in sdc.states])
    # sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)

    for i in range(len(f)):
        x = np.array([state.center_x for state in f[i].states])
        y = np.array([state.center_y for state in f[i].states])
        # pos = np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1)], axis=-1)
        l = np.array([[state.length] for state in f[i].states])[:, 0]
        w = np.array([[state.width] for state in f[i].states])[:, 0]
        head = np.array([state.heading for state in f[i].states])
        vx = np.array([state.velocity_x for state in f[i].states])
        vy = np.array([state.velocity_y for state in f[i].states])
        valid = np.array([[state.valid] for state in f[i].states])[:, 0]
        t = np.repeat(f[i].object_type, len(valid))
        agents[i] = np.stack((x, y, vx, vy, head, l, w, t, valid), axis=-1)[:BATCH_SIZE]
        # agents[i] = np.concatenate((pos, velocity, head, l, w, t, valid), axis=-1)[:BATCH_SIZE]

    ego = agents[[sdc_index]]
    others = np.delete(agents, sdc_index, axis=0)
    # others = np.transpose(others,[1,0,2])
    # others[abs(others[..., 0] > 80), -1] = 0
    # others[abs(others[..., 1] > 80), -1] = 0
    all_agent = np.concatenate([ego, others], axis=0)
    return all_agent.swapaxes(0, 1)


def extract_dynamic(f):
    # dynamics = np.zeros([BATCH_SIZE, 32, 6])
    dynamics = []
    # time_sample = min(int(len(sdc.states)/BATCH_SIZE), TIME_SAMPLE)
    # sdc_x = np.array([state.center_x for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    # sdc_y = np.array([state.center_y for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    # sdc_yaw = np.array([state.heading for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    # sdc_x = np.array([state.center_x for state in sdc.states])[:BATCH_SIZE]
    # sdc_y = np.array([state.center_y for state in sdc.states])[:BATCH_SIZE]
    # sdc_yaw = np.array([state.heading for state in sdc.states])[:BATCH_SIZE]
    # sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)

    for i in range(BATCH_SIZE):
        # states = f[i * time_sample].lane_states
        states = f[i].lane_states
        traf_list = []
        for j in range(len(states)):
            traf = np.zeros(6)
            traf[0] = states[j].lane
            traf[1:4] = np.array([[states[j].stop_point.x, states[j].stop_point.y, 0]])
            if states[j].state in [1, 4, 7]:
                state_ = 1  # stop
            elif states[j].state in [2, 5, 8]:
                state_ = 2  # caution
            elif states[j].state in [3, 6]:
                state_ = 3  # go
            else:
                state_ = 0  # unknown
            traf[4] = state_
            traf[5] = 1 if states[j].state else 0
            traf_list.append(traf)
        dynamics.append(traf_list)
    return dynamics


def extract_poly(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]
    coord = np.stack((x, y, z), axis=1)

    return coord


def down_sampling(line, type=0):
    # if is center lane
    point_num = len(line)

    ret = []

    if point_num < SAMPLE_NUM or type == 1:
        for i in range(0, point_num):
            ret.append(line[i])
    else:
        for i in range(0, point_num, SAMPLE_NUM):
            ret.append(line[i])

    return ret


def extract_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype='int64')
    for k in range(len(fb)):
        c = dict()
        c['index'] = [fb[k].lane_start_index, fb[k].lane_end_index]
        c['type'] = RoadLineType(fb[k].boundary_type)
        c['id'] = fb[k].boundary_feature_id
        b.append(c)

    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb['id'] = fb[k].feature_id
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['boundaries'] = extract_boundaries(fb[k].boundaries)
        nb['id'] = fb[k].feature_id
        nbs.append(nb)
    return nbs


def extract_center(f):
    center = {}
    f = f.lane

    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    poly = [np.insert(x, 2, f.type) for x in poly]

    center['interpolating'] = f.interpolating

    center['entry'] = [x for x in f.entry_lanes]

    center['exit'] = [x for x in f.exit_lanes]

    center['left_boundaries'] = extract_boundaries(f.left_boundaries)

    center['right_boundaries'] = extract_boundaries(f.right_boundaries)

    center['left_neighbor'] = extract_neighbors(f.left_neighbors)

    center['right_neighbor'] = extract_neighbors(f.right_neighbors)

    return poly, center


def extract_line(f):
    f = f.road_line
    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    type = f.type + 5
    poly = [np.insert(x, 2, type) for x in poly]
    return poly


def extract_edge(f):
    f = f.road_edge
    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    type = 15 if f.type == 1 else 16
    poly = [np.insert(x, 2, type) for x in poly]

    return poly


def extract_stop(f):
    f = f.stop_sign
    ret = np.array([f.position.x, f.position.y, 17])

    return [ret]


def extract_crosswalk(f):
    f = f.crosswalk
    poly = down_sampling(extract_poly(f.polygon)[:, :2], 1)
    poly = [np.insert(x, 2, 18) for x in poly]
    return poly


def extract_bump(f):
    f = f.speed_bump
    poly = down_sampling(extract_poly(f.polygon)[:, :2], 1)
    poly = [np.insert(x, 2, 19) for x in poly]
    return poly


def extract_map(f):
    maps = []
    center_infos = {}
    # nearbys = dict()
    for i in range(len(f)):
        id = f[i].id

        if f[i].HasField('lane'):
            line, center_info = extract_center(f[i])
            center_infos[id] = center_info

        elif f[i].HasField('road_line'):
            line = extract_line(f[i])

        elif f[i].HasField('road_edge'):
            line = extract_edge(f[i])

        elif f[i].HasField('stop_sign'):
            line = extract_stop(f[i])

        elif f[i].HasField('crosswalk'):
            line = extract_crosswalk(f[i])

        elif f[i].HasField('speed_bump'):
            line = extract_bump(f[i])
        else:
            continue

        line = [np.insert(x, 3, id) for x in line]
        maps = maps + line

    return np.array(maps), center_infos

def yaw_to_y(angles):
    ret = []
    for angle in angles:
        angle = trans_angle(angle)
        angle_to_y = angle - np.pi / 2
        angle_to_y = -1 * angle_to_y
        ret.append(angle_to_y)
    return np.array(ret)

def transform_coordinate_map(map, sdc):
    """
    Every frame is different
    """
    time_sample = min(int(len(sdc.states) / BATCH_SIZE), TIME_SAMPLE)
    sdc_x = np.array([state.center_x for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_y = np.array([state.center_y for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_yaw = np.array([state.heading for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
    pos = np.stack([sdc_x, sdc_y], axis=-1)
    ret = np.zeros(shape=(BATCH_SIZE, *map.shape))
    for i in range(BATCH_SIZE):
        ret[i] = map
        ret[i][..., :2] = transform_coord(ret[i][..., :2] - pos[i], np.expand_dims(sdc_theta[i], -1))

    # ret[abs(ret[:, :, :, 1]) > 80,-1] =0
    # ret[abs(ret[:, :, :, 2]) > 80, -1] = 0
    valid_ret = np.sum(ret[..., -1], -1)
    lane_mask = valid_ret.astype(bool)
    ret[ret[..., -1] == 0, :] = 0

    return ret, lane_mask


def add_traff_to_lane(scene):
    traf = scene['traf_p_c_f']
    lane = scene['lane']
    traf_buff = np.zeros([*lane.shape[:2]])
    for i in range(BATCH_SIZE):
        lane_i_id = lane[i, :, 0, -1]
        for a_traf in traf[i]:
            lane_id = a_traf[0]
            state = a_traf[-2]
            lane_idx = np.where(lane_i_id == lane_id)
            traf_buff[i, lane_idx] = state
    return traf_buff


def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)


def extract_width(map, polyline, boundary):
    l_width = np.zeros(polyline.shape[0])
    for b in boundary:
        idx = map[:, -1] == b['id']
        b_polyline = map[idx][:, :2]

        start_p = polyline[b['index'][0]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = b['index'][1] - b['index'][0]
        end_index = min(start_index + seg_len, b_polyline.shape[0] - 1)
        leng = min(end_index - start_index, b['index'][1] - b['index'][0]) + 1
        self_range = range(b['index'][0], b['index'][0] + leng)
        bound_range = range(start_index, start_index + leng)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist
    return l_width


def compute_width(scene):
    lane = scene['unsampled_lane']
    lane_id = np.unique(lane[..., -1]).astype(int)
    center_infos = scene['center_info']

    for id in lane_id:
        if not id in center_infos.keys():
            continue
        id_set = lane[..., -1] == id
        points = lane[id_set]

        width = np.zeros((points.shape[0], 2))

        width[:, 0] = extract_width(lane, points[:, :2], center_infos[id]['left_boundaries'])
        width[:, 1] = extract_width(lane, points[:, :2], center_infos[id]['right_boundaries'])

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]

        center_infos[id]['width'] = width
    return

if __name__ == "__main__":
    file_path = '0.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print(type(data))
        print(data.keys())
        print("id: ",data['id'])
        print("all_agent: ", data['all_agent'][0][0])

        print("traffic_light situation: ", len(data['traffic_light']))
        print("traffic_light: ", data['traffic_light'])

        print("lane number: ", len(data['lane']))
        print("lane: ", data['lane'][0])
        print("lane: ", data['lane'][1])
        print("lane: ", data['lane'][15])



    # dataset = tf.data.TFRecordDataset("uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000", compression_type='')
    # scenario = scenario_pb2.Scenario()
    # counter = 0
    # for j, data in enumerate(dataset.as_numpy_iterator()):
    #     scenario.ParseFromString(data)
    #     scene = dict()
    #
    #     scene['id'] = scenario.scenario_id # ID of the scenario
    #
    #     sdc_index = scenario.sdc_track_index #自动驾驶车自身对应tracks数组中的ID
    #     scene['all_agent'] = extract_tracks(scenario.tracks, sdc_index)
    #
    #
    #     scene['traffic_light'] = extract_dynamic(scenario.dynamic_map_states)
    #
    #     SAMPLE_NUM = 10
    #     scene['lane'], scene['center_info'] = extract_map(scenario.map_features)
    #
    #     SAMPLE_NUM = 10e9
    #     scene['unsampled_lane'], _ = extract_map(scenario.map_features)
    #     compute_width(scene)
    #     counter += 1
    #     # print("Agent shape: ", scene['all_agent'].shape)
    #
    #     # print(scenario.scenario_id)
    #     # print(scenario.sdc_track_index)
    #     # print(scenario.tracks)
    #     # print(type(scenario.tracks))
    #     # print(scenario.dynamic_map_states)
    #     # print(scenario.scenario_id)
    #     # print(scenario.timestamps_seconds)
    #     # print(scenario.tracks[0].id) # 每个track代表一段时间内对象的状态。
    #     # print(len(scenario.map_features))
    #     # print("Agent shape: ", scene['all_agent'].shape)
    #
    #
    # print("Counter:", counter)
    #
    # print("Agent shape: ",scene['all_agent'].shape)
    # print("x, y, vx, vy, heading, l, w, type, valid: ", scene['all_agent'][0][1])
    # print("x, y, vx, vy, heading, l, w, type, valid: ", scene['all_agent'][1][1])
    #
    # print("traffic_light shape: ", len(scene['traffic_light']))
    # print("lane, stopx, stopy, 0, light state, same as light state: ", scene['traffic_light'][0][0])
    #
    # print("lane shape: ", len(scene['lane']))
    # print("lane: position + map ID", scene['lane'][0])
    #
    #
    # print(type(scene['center_info']))





