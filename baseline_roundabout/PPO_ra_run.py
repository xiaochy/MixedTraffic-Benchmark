import argparse
import os
import random
import wandb
import ray
import ast
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from Env import Env
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from core.custom_logger import CustomLoggerCallback
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
    "--lane-entry", type=str, default="", help="Input the lane entry id list"
)

parser.add_argument(
    "--junction-id", type=str, default="", help="Input the junction id list"
)

if __name__ == "__main__":

    args = parser.parse_args()

    ray.init(num_gpus=1, num_cpus=args.num_cpus)

    # 创建虚拟环境
    dummy_env = Env({
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "lane_entry":ast.literal_eval(args.lane_entry),
            "junction_id":ast.literal_eval(args.junction_id),
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
    })

    # 提取观察空间和动作空间
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    # 关闭虚拟环境
    dummy_env.close()
    

    # 初始化 Ray
    #ray.init(ignore_reinit_error=True)

    # 配置 PPO 算法
    #print("init config")
    config = (
        PPOConfig()
        .environment(Env, disable_env_checking=True, env_config={
                "spawn_rl_prob":{},
                "probablity_RL":args.rv_rate,
                "lane_entry":ast.literal_eval(args.lane_entry),
                "junction_id":ast.literal_eval(args.junction_id),
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
        .framework("torch")  # 使用 PyTorch，或 "tf" 选择 TensorFlow
        .training(
            model={
                "fcnet_hiddens": [256, 256],  # 网络结构
                "fcnet_activation": "relu",    # 激活函数
            },
            use_gae=True,  # 使用广义优势估计（GAE）
            lambda_=0.95,  # GAE lambda
            clip_param=0.2,  # PPO剪辑参数
            vf_clip_param=10.0,  # Value function的剪辑范围
            entropy_coeff=0.01,  # Entropy系数，用于激励探索
        )
        .rollouts(num_rollout_workers=1, rollout_fragment_length=200)  # 配置 Rollout Workers
        .resources(num_gpus=1, num_cpus_per_worker=2)  # 资源配置
    )

    # 训练配置
    stop = {
        "training_iteration": args.stop_iters,  # 训练迭代次数
    }

    if args.resume_cp != "":
        resume = True
        resume_cp = args.resume_cp
    else:
        resume = False
    save_path = args.save_dir

    # 实例化 Trainer
    trainer = config.build()

    # 训练
    for i in range(stop["training_iteration"]):
        if resume:
            trainer.restore(resume_cp)
            resume = False
        results = trainer.train()
    save_result = trainer.save(save_path)
        #print(f"Iteration {i}: reward mean = {result['episode_reward_mean']}")

    # # 保存模型
    # checkpoint_path = trainer.save()
    # print(f"Model saved at {checkpoint_path}")

    # 关闭 Ray
    ray.shutdown()
