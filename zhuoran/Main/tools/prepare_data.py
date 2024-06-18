import copy
import os
import random
import numpy as np
import torch
from shapely.geometry import Polygon
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import os
import pickle

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

LANE_SAMPLE = 10
RANGE = 50


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_angle(angle):
    if isinstance(angle, torch.Tensor):
        while not torch.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not torch.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2
        return angle

    else:
        while not np.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not np.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2

        return angle


def cal_rel_dir(dir1, dir2):
    dist = dir1 - dir2

    while not np.all(dist >= 0):
        dist[dist < 0] += np.pi * 2
    while not np.all(dist < np.pi * 2):
        dist[dist >= np.pi * 2] -= np.pi * 2

    dist[dist > np.pi] -= np.pi * 2
    return dist


def wash(batch):
    for key in batch.keys():
        if batch[key].dtype == np.float64:
            batch[key] = batch[key].astype(np.float32)
        if 'mask' in key:
            batch[key] = batch[key].astype(bool)
        if isinstance(batch[key], torch.DoubleTensor):
            batch[key] = batch[key].float()


def process_lane(lane, max_vec, lane_range, offset=-40):
    # dist = lane[..., 0]**2+lane[..., 1]**2
    # idx = np.argsort(dist)
    # lane = lane[idx]

    vec_dim = 6

    lane_point_mask = (abs(lane[..., 0] + offset) < lane_range) * (abs(lane[..., 1]) < lane_range)

    lane_id = np.unique(lane[..., -2]).astype(int)

    vec_list = []
    vec_mask_list = []
    b_s, _, lane_dim = lane.shape

    for id in lane_id:
        id_set = lane[..., -2] == id
        points = lane[id_set].reshape(b_s, -1, lane_dim)
        masks = lane_point_mask[id_set].reshape(b_s, -1)

        vector = np.zeros([b_s, points.shape[1] - 1, vec_dim])
        vector[..., 0:2] = points[:, :-1, :2]
        vector[..., 2:4] = points[:, 1:, :2]
        # id
        # vector[..., 4] = points[:,1:, 3]
        # type
        vector[..., 4] = points[:, 1:, 2]
        # traffic light
        vector[..., 5] = points[:, 1:, 4]
        vec_mask = masks[:, :-1] * masks[:, 1:]
        vector[vec_mask == 0] = 0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)

    vector = np.concatenate(vec_list, axis=1) if vec_list else np.zeros([b_s, 0, vec_dim])
    vector_mask = np.concatenate(vec_mask_list, axis=1) if vec_mask_list else np.zeros([b_s, 0], dtype=bool)

    all_vec = np.zeros([b_s, max_vec, vec_dim])
    all_mask = np.zeros([b_s, max_vec])
    for t in range(b_s):
        mask_t = vector_mask[t]
        vector_t = vector[t][mask_t]

        dist = vector_t[..., 0]**2 + vector_t[..., 1]**2
        idx = np.argsort(dist)
        vector_t = vector_t[idx]
        mask_t = np.ones(vector_t.shape[0])

        vector_t = vector_t[:max_vec]
        mask_t = mask_t[:max_vec]

        vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
        mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
        all_vec[t] = vector_t
        all_mask[t] = mask_t

    return all_vec, all_mask.astype(bool)


def process_map(lane, traf, center_num=384, edge_num=128, lane_range=60, offest=-40):
    lane_with_traf = np.zeros([*lane.shape[:-1], 5])
    lane_with_traf[..., :4] = lane

    lane_id = lane[..., -1]
    b_s = lane_id.shape[0]

    for i in range(b_s):
        traf_t = traf[i]
        lane_id_t = lane_id[i]
        for a_traf in traf_t:
            control_lane_id = a_traf[0]
            state = a_traf[-2]
            lane_idx = np.where(lane_id_t == control_lane_id)
            lane_with_traf[i, lane_idx, -1] = state

    # lane = np.delete(lane_with_traf,-2,axis=-1)
    lane = lane_with_traf
    lane_type = lane[0, :, 2]
    center_1 = lane_type == 1
    center_2 = lane_type == 2
    center_3 = lane_type == 3
    center_ind = center_1 + center_2 + center_3

    boundary_1 = lane_type == 15
    boundary_2 = lane_type == 16
    bound_ind = boundary_1 + boundary_2

    cross_walk = lane_type == 18
    speed_bump = lane_type == 19
    cross_ind = cross_walk + speed_bump

    rest = ~(center_ind + bound_ind + cross_walk + speed_bump + cross_ind)

    cent, cent_mask = process_lane(lane[:, center_ind], center_num, lane_range, offest)
    bound, bound_mask = process_lane(lane[:, bound_ind], edge_num, lane_range, offest)
    cross, cross_mask = process_lane(lane[:, cross_ind], 32, lane_range, offest)
    rest, rest_mask = process_lane(lane[:, rest], 192, lane_range, offest)

    return cent, cent_mask, bound, bound_mask, cross, cross_mask, rest, rest_mask


def rotate(x, y, angle):
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


class WaymoAgent:
    def __init__(self, feature, vec_based_info=None, range=50, max_speed=30, from_inp=False):
        # index of xy,v,lw,yaw,type,valid

        self.RANGE = range
        self.MAX_SPEED = max_speed

        if from_inp:

            self.position = feature[..., :2] * self.RANGE
            self.velocity = feature[..., 2:4] * self.MAX_SPEED
            self.heading = np.arctan2(feature[..., 5], feature[..., 4])[..., np.newaxis]
            self.length_width = feature[..., 6:8]
            type = np.ones_like(self.heading)
            self.feature = np.concatenate(
                [self.position, self.velocity, self.heading, self.length_width, type], axis=-1
            )
            if vec_based_info is not None:
                vec_based_rep = copy.deepcopy(vec_based_info)
                vec_based_rep[..., 5:9] *= self.RANGE
                vec_based_rep[..., 2] *= self.MAX_SPEED
                self.vec_based_info = vec_based_rep

        else:
            self.feature = feature
            self.position = feature[..., :2]
            self.velocity = feature[..., 2:4]
            self.heading = feature[..., [4]]
            self.length_width = feature[..., 5:7]
            self.type = feature[..., [7]]
            self.vec_based_info = vec_based_info

    @staticmethod
    def from_list_to_array(inp_list):
        MAX_AGENT = 32
        agent = np.concatenate([x.get_inp(act=True) for x in inp_list], axis=0)
        agent = agent[:MAX_AGENT]
        agent_num = agent.shape[0]
        agent = np.pad(agent, ([0, MAX_AGENT - agent_num], [0, 0]))
        agent_mask = np.zeros([agent_num])
        agent_mask = np.pad(agent_mask, ([0, MAX_AGENT - agent_num]))
        agent_mask[:agent_num] = 1
        agent_mask = agent_mask.astype(bool)
        return agent, agent_mask

    def get_agent(self, index):
        return WaymoAgent(self.feature[[index]], self.vec_based_info[[index]])

    def get_list(self):
        bs, agent_num, feature_dim = self.feature.shape
        vec_dim = self.vec_based_info.shape[-1]
        feature = self.feature.reshape([-1, feature_dim])
        vec_rep = self.vec_based_info.reshape([-1, vec_dim])
        agent_num = feature.shape[0]
        lis = []
        for i in range(agent_num):
            lis.append(WaymoAgent(feature[[i]], vec_rep[[i]]))
        return lis

    def get_inp(self, act=False, act_inp=False):

        if act:
            return np.concatenate([self.position, self.velocity, self.heading, self.length_width], axis=-1)

        pos = self.position / self.RANGE
        velo = self.velocity / self.MAX_SPEED
        cos_head = np.cos(self.heading)
        sin_head = np.sin(self.heading)

        if act_inp:
            return np.concatenate([pos, velo, cos_head, sin_head, self.length_width], axis=-1)

        vec_based_rep = copy.deepcopy(self.vec_based_info)
        vec_based_rep[..., 5:9] /= self.RANGE
        vec_based_rep[..., 2] /= self.MAX_SPEED
        agent_feat = np.concatenate([pos, velo, cos_head, sin_head, self.length_width, vec_based_rep], axis=-1)
        return agent_feat

    def get_rect(self, pad=0):

        l, w = (self.length_width[..., 0] + pad) / 2, (self.length_width[..., 1] + pad) / 2
        x1, y1 = l, w
        x2, y2 = l, -w

        point1 = rotate(x1, y1, self.heading[..., 0])
        point2 = rotate(x2, y2, self.heading[..., 0])
        center = self.position

        x1, y1 = point1[..., [0]], point1[..., [1]]
        x2, y2 = point2[..., [0]], point2[..., [1]]

        p1 = np.concatenate([center[..., [0]] + x1, center[..., [1]] + y1], axis=-1)
        p2 = np.concatenate([center[..., [0]] + x2, center[..., [1]] + y2], axis=-1)
        p3 = np.concatenate([center[..., [0]] - x1, center[..., [1]] - y1], axis=-1)
        p4 = np.concatenate([center[..., [0]] - x2, center[..., [1]] - y2], axis=-1)

        p1 = p1.reshape(-1, p1.shape[-1])
        p2 = p2.reshape(-1, p1.shape[-1])
        p3 = p3.reshape(-1, p1.shape[-1])
        p4 = p4.reshape(-1, p1.shape[-1])

        agent_num, dim = p1.shape

        rect_list = []
        for i in range(agent_num):
            rect = np.stack([p1[i], p2[i], p3[i], p4[i]])
            rect_list.append(rect)
        return rect_list

    def get_polygon(self):
        rect_list = self.get_rect(pad=0.25)

        poly_list = []
        for i in range(len(rect_list)):
            a = rect_list[i][0]
            b = rect_list[i][1]
            c = rect_list[i][2]
            d = rect_list[i][3]
            poly_list.append(Polygon([a, b, c, d]))

        return poly_list


def get_vec_based_rep(case_info):

    thres = 5
    max_agent_num = 32
    # process future agent

    agent = case_info['agent']
    vectors = case_info["center"]

    agent_mask = case_info['agent_mask']

    vec_x = ((vectors[..., 0] + vectors[..., 2]) / 2)
    vec_y = ((vectors[..., 1] + vectors[..., 3]) / 2)

    agent_x = agent[..., 0]
    agent_y = agent[..., 1]

    b, vec_num = vec_y.shape
    _, agent_num = agent_x.shape

    vec_x = np.repeat(vec_x[:, np.newaxis], axis=1, repeats=agent_num)
    vec_y = np.repeat(vec_y[:, np.newaxis], axis=1, repeats=agent_num)

    agent_x = np.repeat(agent_x[:, :, np.newaxis], axis=-1, repeats=vec_num)
    agent_y = np.repeat(agent_y[:, :, np.newaxis], axis=-1, repeats=vec_num)

    dist = np.sqrt((vec_x - agent_x)**2 + (vec_y - agent_y)**2)

    cent_mask = np.repeat(case_info['center_mask'][:, np.newaxis], axis=1, repeats=agent_num)
    dist[cent_mask == 0] = 10e5
    vec_index = np.argmin(dist, -1)
    min_dist_to_lane = np.min(dist, -1)
    min_dist_mask = min_dist_to_lane < thres

    selected_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], axis=1)

    vx, vy = agent[..., 2], agent[..., 3]
    v_value = np.sqrt(vx**2 + vy**2)
    low_vel = v_value < 0.1

    dir_v = np.arctan2(vy, vx)
    x1, y1, x2, y2 = selected_vec[..., 0], selected_vec[..., 1], selected_vec[..., 2], selected_vec[..., 3]
    dir = np.arctan2(y2 - y1, x2 - x1)
    agent_dir = agent[..., 4]

    v_relative_dir = cal_rel_dir(dir_v, agent_dir)
    relative_dir = cal_rel_dir(agent_dir, dir)

    v_relative_dir[low_vel] = 0

    v_dir_mask = abs(v_relative_dir) < np.pi / 6
    dir_mask = abs(relative_dir) < np.pi / 4

    agent_x = agent[..., 0]
    agent_y = agent[..., 1]
    vec_x = (x1 + x2) / 2
    vec_y = (y1 + y2) / 2

    cent_to_agent_x = agent_x - vec_x
    cent_to_agent_y = agent_y - vec_y

    coord = rotate(cent_to_agent_x, cent_to_agent_y, np.pi / 2 - dir)

    vec_len = np.clip(np.sqrt(np.square(y2 - y1) + np.square(x1 - x2)), a_min=4.5, a_max=5.5)

    lat_perc = np.clip(coord[..., 0], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
    long_perc = np.clip(coord[..., 1], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len

    total_mask = min_dist_mask * agent_mask * v_dir_mask * dir_mask
    total_mask[:, 0] = 1
    total_mask = total_mask.astype(bool)

    b_s, agent_num, agent_dim = agent.shape
    agent_ = np.zeros([b_s, max_agent_num, agent_dim])
    agent_mask_ = np.zeros([b_s, max_agent_num]).astype(bool)

    the_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], 1)
    # 0: vec_index
    # 1-2 long and lat percent
    # 3-5 velocity and direction
    # 6-9 lane vector
    # 10-11 lane type and traff state
    info = np.concatenate(
        [
            vec_index[..., np.newaxis], long_perc[..., np.newaxis], lat_perc[..., np.newaxis], v_value[..., np.newaxis],
            v_relative_dir[..., np.newaxis], relative_dir[..., np.newaxis], the_vec
        ], -1
    )

    info_ = np.zeros([b_s, max_agent_num, info.shape[-1]])

    for i in range(agent.shape[0]):
        agent_i = agent[i][total_mask[i]]
        info_i = info[i][total_mask[i]]

        agent_i = agent_i[:max_agent_num]
        info_i = info_i[:max_agent_num]

        valid_num = agent_i.shape[0]
        agent_i = np.pad(agent_i, [[0, max_agent_num - agent_i.shape[0]], [0, 0]])
        info_i = np.pad(info_i, [[0, max_agent_num - info_i.shape[0]], [0, 0]])

        agent_[i] = agent_i
        info_[i] = info_i
        agent_mask_[i, :valid_num] = True

    # case_info['vec_index'] = info[...,0].astype(int)
    # case_info['relative_dir'] = info[..., 1]
    # case_info['long_perc'] = info[..., 2]
    # case_info['lat_perc'] = info[..., 3]
    # case_info['v_value'] = info[..., 4]
    # case_info['v_dir'] = info[..., 5]

    case_info['vec_based_rep'] = info_[..., 1:]
    case_info['agent_vec_index'] = info_[..., 0].astype(int)
    case_info['agent_mask'] = agent_mask_
    case_info["agent"] = agent_

    return


def transform_coordinate_map(data):
    """
    Every frame is different
    """
    timestep = data['all_agent'].shape[0]

    # sdc_theta = data['sdc_theta'][:,np.newaxis]
    ego = data['all_agent'][:, 0]
    pos = ego[:, [0, 1]][:, np.newaxis]

    lane = data['lane'][np.newaxis]
    lane = np.repeat(lane, timestep, axis=0)
    lane[..., :2] -= pos

    x = lane[..., 0]
    y = lane[..., 1]
    ego_heading = ego[:, [4]]
    lane[..., :2] = rotate(x, y, -ego_heading)

    unsampled_lane = data['unsampled_lane'][np.newaxis]
    unsampled_lane = np.repeat(unsampled_lane, timestep, axis=0)
    unsampled_lane[..., :2] -= pos

    x = unsampled_lane[..., 0]
    y = unsampled_lane[..., 1]
    ego_heading = ego[:, [4]]
    unsampled_lane[..., :2] = rotate(x, y, -ego_heading)
    return lane, unsampled_lane[0]


def process_agent(agent, sort_agent):

    ego = agent[:, 0]

    ego_pos = copy.deepcopy(ego[:, :2])[:, np.newaxis]
    ego_heading = ego[:, [4]]

    agent[..., :2] -= ego_pos
    agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
    agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
    agent[..., 4] -= ego_heading

    agent_mask = agent[..., -1]
    agent_type_mask = agent[..., -2] == 1
    agent_range_mask = (abs(agent[..., 0]) < RANGE) * (abs(agent[..., 1]) < RANGE)
    mask = agent_mask * agent_type_mask * agent_range_mask

    bs, agent_num, _ = agent.shape
    sorted_agent = np.zeros_like(agent)
    sorted_mask = np.zeros_like(agent_mask).astype(bool)
    sorted_agent[:, 0] = agent[:, 0]
    sorted_mask[:, 0] = True
    for i in range(bs):
        xy = copy.deepcopy(agent[i, 1:, :2])
        agent_i = copy.deepcopy(agent[i, 1:])
        mask_i = mask[i, 1:]

        # put invalid agent to the right down side
        xy[mask_i == False, 0] = 10e8
        xy[mask_i == False, 1] = -10e8

        raster = np.floor(xy / 0.25)
        raster = np.concatenate([raster, agent_i, mask_i[:, np.newaxis]], -1)
        y_index = np.argsort(-raster[:, 1])
        raster = raster[y_index]
        y_set = np.unique(raster[:, 1])[::-1]
        for y in y_set:
            ind = np.argwhere(raster[:, 1] == y)[:, 0]
            ys = raster[ind]
            x_index = np.argsort(ys[:, 0])
            raster[ind] = ys[x_index]
        # scene = np.delete(raster, [0, 1], axis=-1)
        sorted_agent[i, 1:] = raster[..., 2:-1]
        sorted_mask[i, 1:] = raster[..., -1]

    if sort_agent:
        return sorted_agent[..., :-1], sorted_mask
    else:
        agent_nums = np.sum(sorted_mask, axis=-1)
        for i in range(sorted_agent.shape[0]):
            agent_num = int(agent_nums[i])
            permut_idx = np.random.permutation(np.arange(1, agent_num)) - 1
            sorted_agent[i, 1:agent_num] = sorted_agent[i, 1:agent_num][permut_idx]
        return sorted_agent[..., :-1], sorted_mask


def get_gt(case_info):

    # 0: vec_index
    # 1-2 long and lat percent
    # 3-5 speed, angle between velocity and car heading, angle between car heading and lane vector
    # 6-9 lane vector
    # 10-11 lane type and traff state
    center_num = case_info['center'].shape[1]
    lane_inp = case_info['lane_inp'][:, :center_num]
    agent_vec_index = case_info['agent_vec_index']
    vec_based_rep = case_info['vec_based_rep']
    bbox = case_info['agent'][..., 5:7]

    b, lane_num, _ = lane_inp.shape
    gt_distribution = np.zeros([b, lane_num])
    gt_vec_based_coord = np.zeros([b, lane_num, 5])
    gt_bbox = np.zeros([b, lane_num, 2])
    for i in range(b):
        mask = case_info['agent_mask'][i].sum()
        index = agent_vec_index[i].astype(int)
        gt_distribution[i][index[:mask]] = 1
        gt_vec_based_coord[i, index] = vec_based_rep[i, :, :5]
        gt_bbox[i, index] = bbox[i]
    case_info['gt_bbox'] = gt_bbox
    case_info['gt_distribution'] = gt_distribution
    case_info['gt_long_lat'] = gt_vec_based_coord[..., :2]
    case_info['gt_speed'] = gt_vec_based_coord[..., 2]
    case_info['gt_vel_heading'] = gt_vec_based_coord[..., 3]
    case_info['gt_heading'] = gt_vec_based_coord[..., 4]


def _process_map_inp(case_info):
    center = copy.deepcopy(case_info['center'])
    center[..., :4] /= RANGE
    edge = copy.deepcopy(case_info['bound'])
    edge[..., :4] /= RANGE
    cross = copy.deepcopy(case_info['cross'])
    cross[..., :4] /= RANGE
    rest = copy.deepcopy(case_info['rest'])
    rest[..., :4] /= RANGE

    case_info['lane_inp'] = np.concatenate([center, edge, cross, rest], axis=1)
    case_info['lane_mask'] = np.concatenate(
        [case_info['center_mask'], case_info['bound_mask'], case_info['cross_mask'], case_info['rest_mask']], axis=1
    )
    return


def process_data_to_internal_format(data):
    case_info = {}
    gap = 20

    other = {}

    other['traf'] = data['traffic_light']

    agent = copy.deepcopy(data['all_agent'])
    data['all_agent'] = data['all_agent'][0:-1:gap]
    data['lane'], other['unsampled_lane'] = transform_coordinate_map(data)
    data['traffic_light'] = data['traffic_light'][0:-1:gap]

    other['lane'] = data['lane'][0]

    # transform agent coordinate
    ego = agent[:, 0]
    ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
    ego_heading = ego[[0], [4]]
    agent[..., :2] -= ego_pos
    agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
    agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
    agent[..., 4] -= ego_heading
    agent_mask = agent[..., -1]
    agent_type_mask = agent[..., -2]
    agent_range_mask = (abs(agent[..., 0]) < RANGE) * (abs(agent[..., 1]) < RANGE)
    mask = agent_mask * agent_type_mask * agent_range_mask

    agent = WaymoAgent(agent)
    other['gt_agent'] = agent.get_inp(act=True)
    other['gt_agent_mask'] = mask

    # process agent and lane data
    case_info["agent"], case_info["agent_mask"] = process_agent(data['all_agent'], False)
    case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
    case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
        data['lane'], data['traffic_light'], lane_range=RANGE, offest=0)

    # get vector-based representation
    get_vec_based_rep(case_info)

    agent = WaymoAgent(case_info['agent'], case_info['vec_based_rep'])

    case_info['agent_feat'] = agent.get_inp()

    _process_map_inp(case_info)

    get_gt(case_info)

    case_num = case_info['agent'].shape[0]
    case_list = []
    for i in range(case_num):
        dic = {}
        for k, v in case_info.items():
            dic[k] = v[i]
        case_list.append(dic)

    case_list[0]['other'] = other

    return case_list




agent_feature_map = []
agent_feature_graph = []
agent_label = []

class MCG(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MCG, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_gate = nn.Linear(input_dim, output_dim)
        self.fc_context = nn.Linear(input_dim, output_dim)
        self.output_context = nn.Linear(output_dim, output_dim)  # Added output context layer

    def forward(self, input_vectors, context_vector):
        gate = torch.sigmoid(self.fc_gate(input_vectors))
        context = torch.tanh(self.fc_context(input_vectors))
        gated_context = gate * context
        new_context_vector = context_vector + gated_context.sum(dim=0)
        new_context_vector = self.output_context(new_context_vector).squeeze(0)  # Squeeze context vector
        new_output_vectors = gated_context + new_context_vector

        return new_output_vectors, new_context_vector

# Example usage
input_dim = 1024
output_dim = 1024
num_blocks = 5
num_groups = 32

base_mcg = MCG(input_dim, output_dim)
torch.save(base_mcg.state_dict(), '/content/drive/MyDrive/Research/Traffic/Model/base_mcg_1024.pth')

# ----

time_point = 20 * 5 # Gap * n

for file_name in range(0,1):

  file_path = '/content/drive/MyDrive/Research/Traffic/Data/processed/'+ str(file_name) +'.pkl'
  with open(file_path, 'rb') as file:
    data = pickle.load(file)

  internal_data = process_data_to_internal_format(data)

  # Assuming internal_data is a dictionary containing your data
  combine_agent = []

  for i in range(len(internal_data)):
      combine_agent_row = []  # Initialize a list for the current row
      for j in range(len(internal_data[i]['vec_based_rep'])):
          # Combine the values and append them to the current row
          combined_value = np.concatenate((internal_data[i]['vec_based_rep'][j], internal_data[i]['agent'][j]))
          combine_agent_row.append(combined_value)
      combine_agent.append(combine_agent_row)

  #print(len(combine_agent[0][0]))

  # Assuming combine_agent is a list of lists, where each element is also a list of lists

  for i in range(len(combine_agent)):
    # Get the first numpy array of the current element
    for j in range(len(combine_agent[i])):
      array = combine_agent[i][j]

      # Check if the array needs to be extended
      if array.size < 1024:
          # Calculate how many zeros are needed to make the length 1024
          zeros_needed = 1024 - array.size

          # Create an array of zeros of the required length
          zeros_array = np.zeros(zeros_needed)

          # Concatenate the original array with the zeros array
          combine_agent[i][j] = np.concatenate((array, zeros_array))

  #print(len(combine_agent[0][0]))


  # Initialize the input context vector as 0
  context_vector = torch.zeros(1, output_dim)

  # Input data
  input_data = torch.tensor(combine_agent[:5], dtype=torch.float)

  # Apply MCG blocks sequentially
  base_mcg = MCG(input_dim, output_dim)
  base_mcg.load_state_dict(torch.load('/content/drive/MyDrive/Research/Traffic/Model/base_mcg_1024.pth'))

  for i in range(5):
      #output_data, context_vector = mcg_block(input_data[i], context_vector)
      output_data, context_vector = base_mcg(input_data[i], context_vector)
      input_data[i] = output_data


  with open(file_path, 'rb') as file:
    data = pickle.load(file)

  vehicle_agent = context_vector
  vehicle_agent_label = data['all_agent'][time_point][0][:2]

  ego = data['all_agent'][time_point-20][0]
  other_cars = list()
  for car in data['all_agent'][time_point-20][1:]:
    other_cars.append(car[:2])


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


  node_embed = []
  node_label = []
  node_embed.append(data['all_agent'][time_point-20][0][:2])
  node_label.append(data['all_agent'][time_point][0][:2])
  for i in closest_indices:
    node_embed.append(data['all_agent'][time_point-20][i][:2])

  node_embed = [arr.tolist() for arr in node_embed]

  agent_feature_map.append(vehicle_agent.detach().numpy().tolist())
  agent_feature_graph.append(node_embed)
  agent_label.append(vehicle_agent_label.tolist())

print(len(agent_feature_map))
print(len(agent_feature_graph))
print(len(agent_label))

agent_feature_map_test = agent_feature_map[:10]
agent_feature_graph_test = agent_feature_graph[:10]
agent_label_test = agent_label[:10]

agent_feature_map = agent_feature_map[10:]
agent_feature_graph = agent_feature_graph[10:]
agent_label = agent_label[10:]

print(len(agent_feature_map))
print(len(agent_feature_graph))
print(len(agent_label))