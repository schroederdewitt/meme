import gym

env = gym.make("PongNoFrameskip-v4")

import pickle
import torch as th
with open("pong_obsspace.pt", "wb") as f:
    th.save(env.observation_space, f)

a = 5