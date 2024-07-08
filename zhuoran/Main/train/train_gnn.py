import pytorch_lightning as pl
import torch
import torch.utils.data as data
# General config
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


from trafficgen.act.utils.act_dataset import actDataset
from trafficgen.utils.config import get_parsed_args
from trafficgen.utils.config import load_config_act
from trafficgen.utils.typedef import AgentType, RoadEdgeType, RoadLineType

args = get_parsed_args()
cfg = load_config_act(args.config)

train_set = actDataset(cfg)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

print(train_set)