from trafficgen.traffic_generator.traffic_generator import TrafficGen
from trafficgen.traffic_generator.utils.utils import get_parsed_args
from trafficgen.utils.config import load_config_init

# Please keep this line here:
from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType

if __name__ == "__main__":
    args = get_parsed_args() # 解析命令行参数, return args对象

    # 加载的配置文件内容 cfg。
    cfg = load_config_init(args.config) # args.config = "trafficgen/init/configs/local.yaml"

    print('loading checkpoint...')
    trafficgen = TrafficGen(cfg)
    print('Complete.\n')

    trafficgen.generate_scenarios(gif=args.gif, save_metadrive=args.save_metadrive)
