import argparse
import os
import random
import wandb
import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig, DQNTorchPolicy
from Env import Env
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
    TorchSharedWeightsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from core.custom_logger import CustomLoggerCallback
from ray.rllib.algorithms.sac import SACTorchPolicy, SACConfig
import cProfile
tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=2)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)

parser.add_argument(
    "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
)

parser.add_argument(
    "--wandb-id", type=str, default="", help="Input the wandb id"
)

parser.add_argument(
    "--cfg", type=str, default="", help="Input the path of the cfg file"
)

parser.add_argument(
    "--map-xml", type=str, default="", help="Input the path of the net.xml file"
)

parser.add_argument(
    "--wandb-name", type=str, default="", help="Input the name of the wandb run"
)

parser.add_argument(
    "--resume-cp", type=str, default="", help="Input the resumed checkpoint dir"
)



if __name__ == "__main__":
    args = parser.parse_args()
    #args.num_cpus = 0

    #ray.init(num_gpus=1, num_cpus=args.num_cpus)
    ray.init(num_gpus=1, num_cpus=args.num_cpus)

    # 创建虚拟环境
    dummy_env = Env({
           "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "wandb_id":args.wandb_id,
            "wandb_name":args.wandb_name,
            
            "cfg": args.cfg,
            "render": False,
            "map_xml": args.map_xml,

            "max_episode_steps":3000,
            "conflict_mechanism":'flexible',
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0 
            }
    })

    # 提取观察空间和动作空间
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    # 关闭虚拟环境
    dummy_env.close()

    # 定义策略
    policy = {
        "shared_policy": (
            SACTorchPolicy,
            obs_space,
            act_space,
            None
        )
    }

    # 定义策略映射函数
    policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy"

    # 修改算法配置
    config = (
        SACConfig()
        .environment(Env, disable_env_checking=True, env_config={
            "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "wandb_id":args.wandb_id,
            "wandb_name":args.wandb_name,

            "cfg": args.cfg,
            "render": False,
            "map_xml": args.map_xml,

            "max_episode_steps":3000,
            "conflict_mechanism":'flexible',
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0 
            }
        }, 
        auto_wrap_old_gym_envs=False)
        .framework(args.framework)
        .training(
            twin_q = True,
            q_model_config = {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            policy_model_config = {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            replay_buffer_config={
                'type':'MultiAgentPrioritizedReplayBuffer',
                'prioritized_replay_alpha':0.5,
                'capacity':50000,
            }
        )
        .rollouts(num_rollout_workers=args.num_cpus-1, rollout_fragment_length="auto")
        .multi_agent(policies=policy, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=1, num_cpus_per_worker=2)
    )

    # 设置停止条件
    # stop = {
    #     "training_iteration": args.stop_iters,
    # }

    if args.resume_cp != "":
        resume = True
        resume_path = args.resume_cp
    else:
        resume = False


    # 使用SAC算法进行训练
    trainer = config.build()
    if resume:
        trainer.restore(resume_path)
    policy = trainer.get_policy("shared_policy")
    #print("policy = ", policy)

    # Create environment for simulation
    env = Env({
        "junction_list":['229','499','332','334'],
        "spawn_rl_prob":{},
        "probablity_RL":args.rv_rate,
        "wandb_id":args.wandb_id,
        "wandb_name":args.wandb_name,

        "cfg": args.cfg,
        "render": False,
        "map_xml": args.map_xml,
        
        "max_episode_steps":3000,
        "conflict_mechanism":'flexible',
        "traffic_light_program":{
            "disable_state":'G',
            "disable_light_start":0 
        }
    })

    # Run simulation
    dones = truncated = {}
    dones['__all__'] = truncated['__all__'] = False


    obs, info = env.reset()

    for i in range(5):
        while not dones['__all__'] and not truncated['__all__']:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = policy.compute_single_action(agent_obs,policy_id="shared_policy")
            obs, reward, dones, truncated, info = env.step(actions)
            for key, done in dones.items():
                if done:
                    obs.pop(key)
            if dones['__all__']:
                obs, info = env.reset()

    env.close()

    # 关闭Ray
    ray.shutdown()




