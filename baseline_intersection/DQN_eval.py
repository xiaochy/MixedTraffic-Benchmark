# from ray.rllib.algorithms.algorithm import Algorithm



# from ray.rllib.algorithms.ppo import PPOConfig
# import argparse
# import os
# import random
# import ast
# import ray
# from ray import air, tune
# from ray.rllib.algorithms.dqn import DQNConfig, DQNTorchPolicy
# from Env import Env
# from ray.rllib.examples.models.shared_weights_model import (
#     SharedWeightsModel1,
#     SharedWeightsModel2,
#     TF2SharedWeightsModel,
#     TorchSharedWeightsModel,
# )
# from ray.rllib.models import ModelCatalog
# from ray.rllib.policy.policy import PolicySpec
# from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.utils.test_utils import check_learning_achieved

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--run", type=str, default="DQN", help="The RLlib-registered algorithm to use."
# )
# parser.add_argument("--num-cpus", type=int, default=1)

# parser.add_argument(
#     "--stop-timesteps",
#     type=int,
#     default=1000,
#     help="Number of timesteps to test.",
# )

# parser.add_argument(
#     "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
# )

# parser.add_argument(
#     "--wandb-id", type=str, default="", help="Input the wandb id"
# )

# parser.add_argument(
#     "--wandb-name", type=str, default="", help="Input the name of the wandb run"
# )

# parser.add_argument(
#     "--cfg", type=str, default="", help="Input the path of the cfg file"
# )

# parser.add_argument(
#     "--map-xml", type=str, default="", help="Input the path of the net.xml file"
# )

# parser.add_argument(
#     "--resume-cp", type=str, default="", help="Input the resumed checkpoint dir"
# )

# parser.add_argument(
#     "--junction-list", type=str, default="", help="Input the junction id list"
# )

# if __name__ == "__main__":
#     args = parser.parse_args()

#     # ray.init(num_cpus=args.num_cpus or None)
#     ray.init(local_mode= True)

#     rv_rate = args.rv_rate

#     checkpoint_path = args.resume_cp
#     algo = Algorithm.from_checkpoint(checkpoint_path)
    
#     ## TODO map xml could be parsed from sumocfg file
#     env = Env(config = {
#             #"junction_list":['229','499','332','334'],
#             "junction_list":ast.literal_eval(args.junction_id),
#             "spawn_rl_prob":{},
#             "probablity_RL":rv_rate,
#              "wandb_id": args.wandb_id,
#             "wandb_name": args.wandb_name,
#             "cfg":args.cfg,
#             "render":False,
#             "map_xml":args.map_xml,
#             "max_episode_steps":3000,
#             "conflict_mechanism":'off',
#             "traffic_light_program":{
#                 "disable_state":'G',
#                 "disable_light_start":0
#             }
#     }.environment(disable_env_checking=True))
#     #env = Env(config)

#     episode_reward = 0
#     dones = truncated = {}
#     dones['__all__'] = truncated['__all__'] = False


#     obs, info = env.reset()

#     while not dones['__all__'] and not truncated['__all__']:
#         actions = {}
#         for agent_id, agent_obs in obs.items():
#             actions[agent_id] = algo.compute_single_action(agent_obs, explore=args.explore_during_inference ,policy_id="shared_policy")
#         obs, reward, dones, truncated, info = env.step(actions)
#         for key, done in dones.items():
#             if done:
#                 obs.pop(key)
#         if dones['__all__']:
#             obs, info = env.reset()
#             num_episodes += 1
    
#     env.monitor.evaluate()
#     save_path = args.save_dir+'/'+st(args.rv_rate)+'log.pkl'
#     env.monitor.evaluate()
#     env.monitor.save_to_pickle(file_name = save_path)
#     algo.stop()

#     ray.shutdown()

import argparse
import os
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig, DQNTorchPolicy
from Env import Env
from ray.rllib.policy.policy import PolicySpec
import ast
#print("enter!!!!!!!!!!!!!!!!!!!!!")

parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--eval-iters", type=int, default=100, help="Number of iterations for evaluation."
)
parser.add_argument(
    "--resume-cp", type=str, required=True, help="Path to the checkpoint to evaluate."
)
parser.add_argument(
    "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
)
parser.add_argument(
    "--cfg", type=str, default="", help="Input the path of the cfg file"
)
parser.add_argument(
    "--map-xml", type=str, default="", help="Input the path of the net.xml file"
)
parser.add_argument(
    "--junction-list", type=str, default="", help="Input the junction id list"
)

parser.add_argument(
    "--wandb-id", type=str, default="", help="Input the wandb id"
)

parser.add_argument(
    "--wandb-name", type=str, default="", help="Input the name of the wandb run"
)


if __name__ == "__main__":

    args = parser.parse_args()
    args.num_cpus = 0

    ray.init(num_gpus=1, num_cpus=args.num_cpus)
    #rint("init!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Define the evaluation environment
    dummy_env = Env({
        "junction_list": ast.literal_eval(args.junction_list),
        "spawn_rl_prob":{},
        "probablity_RL":args.rv_rate,
        "cfg":args.cfg, 
        "render":False,
        "map_xml":args.map_xml,
        "wandb_id":args.wandb_id,
        "wandb_name":args.wandb_name,
        "max_episode_steps":3000,
        # "conflict_mechanism":'flexible',
        "conflict_mechanism":'standard',
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
            DQNTorchPolicy,
            obs_space,
            act_space,
            None
        )
    }

    # 定义策略映射函数
    policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy"

    # # Define the policy
    # dummy_env = Env(eval_env_config)
    # obs_space = dummy_env.observation_space
    # act_space = dummy_env.action_space
    # dummy_env.close()

    # policy = {
    #     "shared_policy": (
    #         DQNTorchPolicy,
    #         obs_space,
    #         act_space,
    #         None
    #     )}
    # policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy"
    
    config = (
        DQNConfig()
        .environment(Env, disable_env_checking=True, env_config={
            "junction_list": ast.literal_eval(args.junction_list),
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "cfg":args.cfg, 
            "render":False,
            "map_xml":args.map_xml,
            "wandb_id":args.wandb_id,
            "wandb_name":args.wandb_name,
            "max_episode_steps":3000,
            # "conflict_mechanism":'flexible',
            "conflict_mechanism":'standard',
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
        .multi_agent(policies=policy, policy_mapping_fn=policy_mapping_fn)
        .rollouts(num_rollout_workers=args.num_cpus-1, rollout_fragment_length="auto")
    )
    #print("after config!!!!!!!!!!!!!!!!!!!!")

    # Initialize the trainer with the same configuration
    trainer = config.build()
    #print("after trainer!!!!!!!!!!!!!!!")

    # Restore from the checkpoint
    trainer.restore(args.resume_cp)
    #print("after trainer restore!!!!!!!!!!!!!!!!!!!!!!")

    #print("before policy!!!!!!!!!!!!")
    policy = trainer.get_policy("shared_policy")
    #print("after policy!!!!!!!!!!!!!!!!!!!")

    env = Env({
            "junction_list": ast.literal_eval(args.junction_list),
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "cfg":args.cfg, 
            "render":False,
            "map_xml":args.map_xml,
            "wandb_id":args.wandb_id,
            "wandb_name":args.wandb_name,
            "max_episode_steps":3000,
            # "conflict_mechanism":'flexible',
            "conflict_mechanism":'standard',
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0 
            }
        })


    # Run simulation
    dones = truncated = {}
    dones['__all__'] = truncated['__all__'] = False


    obs, info = env.reset()
    #print("reset the env")
    step=0

    while not dones['__all__'] and not truncated['__all__']:
        if step == 3000:
            break
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = policy.compute_single_action(agent_obs,policy_id="shared_policy")
        obs, reward, dones, truncated, info = env.step(actions)
        for key, done in dones.items():
            if done:
                obs.pop(key)
        if dones['__all__']:
            obs, info = env.reset()
        step += 1
        # print(step)

    env.close()

    # 关闭Ray
    ray.shutdown()

    # # Evaluation loop
    # #for i in range(5):
    # results = trainer.evaluate()
    # print(f"Iteration {i+1}: {results['evaluation']['episode_reward_mean']}")

    # ray.shutdown()
