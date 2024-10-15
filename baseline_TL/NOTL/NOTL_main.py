import argparse
import os, sys
import random
import wandb
sys.path.append(os.getcwd())
from Env import Env
parser = argparse.ArgumentParser()


parser.add_argument(
    "--cfg", type=str, default="", help="Input the path of the cfg file"
)

parser.add_argument(
    "--map-xml", type=str, default="", help="Input the path of the net.xml file"
)

parser.add_argument(
    "--wandb-name",type=str, default="", help="Input the name of this wandb run name"
)

parser.add_argument(
    "--wandb-id",type=str, default="", help="Input the id of this wandb run"
)

if __name__ == "__main__":
    args = parser.parse_args()
    args = parser.parse_args()
    args.cfg = "../../test_files/roundabout/easy/outputs/map-4_dict/map-4.sumo.cfg"
    args.map_xml = "../../test_files/roundabout/easy/outputs/map-4_dict/map-4/map-4.net.xml"
    args.wandb_name = "map_4_easy_NOTL_period=150"

    
    ## TODO map xml could be parsed from sumocfg file
    env = Env({
            "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL":0,  # no traffic light and no rv
      
            "cfg": args.cfg,
            "render":False,
            "map_xml": args.map_xml,
  
            "max_episode_steps":3000,
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0
            },
            "wandb_id":args.wandb_id,
            "wandb_name":args.wandb_name
        })

    episode_reward = 0
    dones = truncated = {}
    dones['__all__'] = truncated['__all__'] = False


    obs, info = env.reset(options={'mode': 'HARD'})
    
    while not dones['__all__'] and not truncated['__all__']:
        actions = {}
        obs, reward, dones, truncated, info = env.step_once(actions)
        for key, done in dones.items():
            if done:
                obs.pop(key)
    
    #env.monitor.evaluate()