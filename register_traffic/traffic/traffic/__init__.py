#import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="traffic/MixTraffic-v0",
    entry_point="traffic.envs:Env",
)