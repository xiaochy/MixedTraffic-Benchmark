import argparse
import os
import random
import wandb
import ray
import ast
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
# , SACTrainer,
import cProfile
tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()


parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=2000, help="Number of iterations to train."
)
parser.add_argument(
    "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
)

parser.add_argument(
    "--resume-cp", type=str, default="", help="Input the path of the latest checkpoint"
)

parser.add_argument(
    "--save-dir", type=str, default="", help="Input the path of the saved checkpoint directory"
)

parser.add_argument(
    "--cfg", type=str, default="", help="Input the path of the cfg file"
)

parser.add_argument(
    "--map-xml", type=str, default="", help="Input the path of the net.xml file"
)

parser.add_argument(
    "--wandb-id", type=str, default="", help="Input the wandb id"
)

parser.add_argument(
    "--save", type=str, default="False", help="Input whether to save this checkpoint"
)


if __name__ == "__main__":


    args = parser.parse_args()
    

    ray.init(num_gpus=1, num_cpus=args.num_cpus)

    # 创建虚拟环境
    dummy_env = Env({
           "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "wandb_id":args.wandb_id,

            #"cfg": "files/outputs/ra_test_dict/ra_test.sumo.cfg",
            "cfg": args.cfg,
            "render":False,
            "map_xml": args.map_xml,
            #"map_xml": "files/outputs/ra_test_dict/ra_test/ra_test.net.xml",

            "max_episode_steps":1000,
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

            "cfg": args.cfg,
            "render":False,
            "map_xml": args.map_xml,

            "max_episode_steps":1000,
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
    stop = {
        "training_iteration": args.stop_iters,
    }
    
    if args.resume_cp != "":
        resume = True
        resume_cp = args.resume_cp
    else:
        resume = False
    save_path = args.save_dir

    # 使用SAC算法进行训练
    trainer = config.build()

    for epoch in range(args.stop_iters):
        if resume:
            trainer.restore(resume_cp)
            resume = False
        results = trainer.train()

    if ast.literal_eval(args.save) == True:
        save_result = trainer.save(save_path)
        #if epoch > 0 and epoch % 10000 == 0: # 1000000 steps  10 checkpoints total
        #     save_result = trainer.save("checkpoints")
        #print("finish epoch = ",epoch)

    # 如果设置了测试标志，检查学习是否已经完成
    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)

    # 关闭Ray
    ray.shutdown()