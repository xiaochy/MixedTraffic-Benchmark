import argparse
import os
import random
import ast
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
    "--junction-list", type=str, default="", help="Input the junction id list"
)

if __name__ == "__main__":

    args = parser.parse_args()

    ray.init(num_gpus=0, num_cpus=args.num_cpus)

    # Create a dummy environment to extract observation and action spaces
    dummy_env = Env({
        "junction_list": ast.literal_eval(args.junction_list),
        "spawn_rl_prob":{},
        "probablity_RL":args.rv_rate,
        "cfg":args.cfg, 
        "render":False,
        "map_xml":args.map_xml,
        "wandb_id":args.wandb_id,
        "max_episode_steps":1000,
        "conflict_mechanism":'flexible',
        "traffic_light_program":{
            "disable_state":'G',
            "disable_light_start":0 
        }
    })
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    dummy_env.close()

    # Define policy
    policy = {
        "shared_policy": (
            DQNTorchPolicy,
            obs_space,
            act_space,
            None
        )}
    policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy"
            
    config = (
        DQNConfig()
        .environment(Env, env_config={
            "junction_list": ast.literal_eval(args.junction_list),
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "cfg":args.cfg, 
            "render":False,
            "map_xml":args.map_xml,
            "wandb_id":args.wandb_id,
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
            num_atoms=51,
            noisy=False,
            hiddens=[512, 512, 512],
            dueling=True,
            double_q=True,
            replay_buffer_config={
                'type':'MultiAgentPrioritizedReplayBuffer',
                'prioritized_replay_alpha':0.5,
                'capacity':50000,
            }
        )
        .rollouts(num_rollout_workers=args.num_cpus-1, rollout_fragment_length="auto")
        .multi_agent(policies=policy, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=0, num_cpus_per_worker=1)
    )

    if args.resume_cp != "":
        resume = True
    else:
        resume = False

    save_path = args.save_dir

    # Initialize the trainer
    trainer = config.build()

    if resume:
        trainer.restore(args.resume_cp)

    # Training loop
    for epoch in range(args.stop_iters):
        results = trainer.train()
        #print("epoch = ",epoch)
    save_result = trainer.save(save_path)    
        # print(f"Epoch: {epoch}, Reward: {results['episode_reward_mean']}")
        
        # if epoch % 10 == 0:  # Save checkpoint every 10 epochs
        #     checkpoint = trainer.save(checkpoint_dir=save_path)
        #     print(f"Checkpoint saved at {checkpoint}")

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
