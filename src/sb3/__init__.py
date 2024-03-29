import os

from sb3.a2c import A2C
from sb3.common.utils import get_system_info
from sb3.ddpg import DDPG
from sb3.dqn import DQN
from sb3.her.her_replay_buffer import HerReplayBuffer
from sb3.ppo import PPO
from sb3.sac import SAC
from sb3.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )


__all__ = [
    "A2C",
    "DDPG",
    "DQN",
    "PPO",
    "SAC",
    "TD3",
    "HerReplayBuffer",
    "get_system_info",
]
