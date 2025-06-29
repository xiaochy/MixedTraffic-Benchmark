import argparse
import os
import random
import wandb
import ray
import ast
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from Env import Env
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from core.custom_logger import CustomLoggerCallback
import cProfile
tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=2)

parser.add_argument(
    "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
)

parser.add_argument(
    "--wandb-id", type=str, default="", help="Input the wandb id"
)

parser.add_argument(
    "--wandb-name", type=str, default="", help="Input the name of the wandb run"
)

parser.add_argument(
    "--cfg", type=str, default="", help="Input the path of the cfg file"
)

parser.add_argument(
    "--map-xml", type=str, default="", help="Input the path of the net.xml file"
)

parser.add_argument(
    "--resume-cp", type=str, default="", help="Input the resumed checkpoint dir"
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

    # Create the environment
    env_config = {
        "junction_list": ['229', '499', '332', '334'],
        "spawn_rl_prob": {},
        "probablity_RL": args.rv_rate,
        "lane_entry": ast.literal_eval(args.lane_entry),
        "junction_id": ast.literal_eval(args.junction_id),
        "wandb_id": args.wandb_id,
        "wandb_name": args.wandb_name,
        "cfg": args.cfg,
        "render": False,
        "map_xml": args.map_xml,
        "max_episode_steps": 3000,
        "conflict_mechanism": 'flexible',
        "traffic_light_program": {
            "disable_state": 'G',
            "disable_light_start": 0
        }
    }

    # Define the algorithm configuration
    config = (
        PPOConfig()
        .environment(Env, env_config=env_config)
        .framework("torch")
        .rollouts(num_rollout_workers=args.num_cpus - 1, rollout_fragment_length="auto")
        .resources(num_gpus=1, num_cpus_per_worker=2)
    )

    if args.resume_cp != "":
        resume = True
        resume_path = args.resume_cp
    else:
        resume = False

    # Build the trainer
    trainer = config.build()
    if resume:
        trainer.restore(resume_path)

    # Run the simulation
    env = Env(env_config)
    obs, info = env.reset()

    #for i in range(5):
    while True:
        action = trainer.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()
            break

    env.close()

    # Shutdown Ray
    ray.shutdown()

