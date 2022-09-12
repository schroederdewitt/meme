from collections import defaultdict
from functools import partial, reduce
import gym
import matplotlib.pyplot as plt
import numpy as np
import operator
import torch as th
from torch import optim, nn
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]

def masked_softmax(x: th.Tensor, mask: th.Tensor, beta: th.Tensor=th.Tensor([1.0]), dim:int=0) -> th.Tensor:
    sc = beta * x
    sc.masked_fill_(mask, -1E20)
    return th.nn.functional.softmax(sc, dim=dim)

from stable_baselines3_lib import pong
class Pong():

    def __init__(self, **kwargs):
        self.reset()
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        if reward != 0.0:
            done = True
        return {"obs_player1": th.from_numpy(obs)}, reward[0], done, info

    def reset(self):
        #obs = self.env.reset()
        self.env = pong.make_atari_env('PongNoFrameskip-v4', n_envs=1,
                                       seed=0, wrapper_kwargs={"noop_max":0})  # wrapper_kwargs={"clip_reward": False, "pong_just_one_step": True},
        self.env = pong.VecFrameStack(self.env, n_stack=4)
        obs = self.env.reset()
        reward = 0.0
        ctr = 0.0

        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"obs_player1": th.from_numpy(obs)}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*6)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        self.env.render()
        pass


class Cartpole():

    def __init__(self, **kwargs):
        from rl_baselines3_zoo_lib import train
        expmanager = train.main(["--algo", "dqn", "--env",
                                 "CartPole-v1",
                                 "-i",
                                 "models/CartPole-v1/CartPole-v1.zip",
                                 "-n", "5000"])
        _ = expmanager.setup_experiment()
        self.env = expmanager.env
        self.env.reset()
        pass

    def step(self, action):
        if th.is_tensor(action):
            action = action.item()

        obs, reward, done, info = self.env.step([action])
        if reward != 0.0:
            self.game_ctr += 1

        return {"obs_player1": th.from_numpy(obs[0])}, reward[0], done, info

    def reset(self):

        obs = self.env.reset()
        self.game_ctr = 0
        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"obs_player1": th.from_numpy(obs[0])}

    def get_available_actions(self):
        return {"player1": th.BoolTensor([True]*self.env.action_space.n)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        self.env.render()
        pass

class PongResRetro():

    def __init__(self, **kwargs):
        self.reset()
        pass

    def step(self, action):
        if th.is_tensor(action):
            action = action.item()
        if action == 0:
            action_trans = 2
        elif action == 1:
            action_trans = 3
        else:
            raise Exception("Unknown action: {}".format(action))
        obs, reward, done, info = self.env.step([action_trans])

        return {"obs_player1": th.from_numpy(obs)}, reward[0], done, info

    def reset(self):

        self.env = pong.make_atari_env('PongNoFrameskip-v4', n_envs=1,
                                       seed=0, wrapper_kwargs={"noop_max":0})
        self.env = pong.VecFrameStack(self.env, n_stack=4)
        obs = self.env.reset()
        reward = 0.0
        ctr = 0.0

        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"obs_player1": th.from_numpy(obs)}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*2)} # Redundant Actions!

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        return self.env.render(mode="rgb_array")


class PongAll():

    def __init__(self, n_games=None, **kwargs):
        self.reset()
        self.n_games = 30 # DBG
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        if reward != 0.0:
            self.end_ctr += 1
            self.reward_ctr += reward[0] # accumulate all reward for the end
            reward = [0.0]
        if self.end_ctr >= self.n_games or done:
            done = True
            reward = [self.reward_ctr]
        return {"obs_player1": th.from_numpy(obs)}, reward[0], done, info

    def reset(self):
        self.reward_ctr = 0.0
        self.end_ctr = 0

        self.env = pong.make_atari_env('PongNoFrameskip-v4', n_envs=1,
                                       seed=0, wrapper_kwargs={"noop_max":0})  # wrapper_kwargs={"clip_reward": False, "pong_just_one_step": True},
        self.env = pong.VecFrameStack(self.env, n_stack=4)
        obs = self.env.reset()
        reward = 0.0
        ctr = 0.0

        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"obs_player1": th.from_numpy(obs)}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*6)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        self.env.render()
        pass

class Breakout():

    def __init__(self, **kwargs):
        self.reset()
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        return {"obs_player1": th.from_numpy(obs)}, reward[0], done, info

    def reset(self):

        self.env = pong.make_atari_env('BreakoutNoFrameskip-v4', n_envs=1,
                                       seed=0, wrapper_kwargs={"noop_max":0})
        self.env = pong.VecFrameStack(self.env, n_stack=4)
        obs = self.env.reset()
        reward = 0.0
        ctr = 0.0

        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"obs_player1": th.from_numpy(obs)}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*4)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        self.env.render()
        pass

class Tennis():

    def __init__(self, **kwargs):
        self.reset()
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        return {"obs_player1": th.from_numpy(obs)}, reward[0], done, info

    def reset(self):

        self.env = pong.make_atari_env('TennisNoFrameskip-v4', n_envs=1,
                                       seed=0, wrapper_kwargs={"noop_max":0})
        self.env = pong.VecFrameStack(self.env, n_stack=4)
        obs = self.env.reset()
        reward = 0.0
        ctr = 0.0

        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"obs_player1": th.from_numpy(obs)}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*18)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        self.env.render()
        pass


class PongPrivateObs():

    def __init__(self, private_obs_dist, token_reward_factor, token_n, private_obs_dim, **kwargs):

        assert callable(private_obs_dist), "Private Obs Dist must be a callable function to sample from!"
        self.private_obs_dist = private_obs_dist
        self.token_reward_factor = token_reward_factor
        self.private_obs_dim = private_obs_dim
        self.token_n = token_n

        from stable_baselines3_lib import pong
        self.env = pong.make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
        self.env = pong.VecFrameStack(self.env, n_stack=4)

    def step(self, action):
        action = action - 1 # because now 0 is no-op
        assert action != -1, "You should never have to call step on no-op in this environment!"

        if not self.done:
            if self.steps > 0:
                self.trajectory["states"].append(self.trajectory["next_states"][-1])
            obs, reward, done, info = self.env.step(action)
            self.trajectory["actions"].append(action)
            self.trajectory["next_states"].append(obs)
            self.trajectory["rewards"].append(reward)
            obs = {"obs_player1": obs, "private_token": self.private_obs, "obs_player2": obs.clone().zero_()}
            self.final_reward = reward
            self.done = done
            return obs, reward, self.done, info
        else:
            reward_p2 = 1.0 if action.to(self.private_obs.device) == self.private_obs else -1.0
            reward = self.final_reward*(1-self.token_reward_factor) + self.token_reward_factor * reward_p2

            obs = {"obs_player1": th.from_numpy(self.get_obs_space()["player1"].low).zero_(),
                   "private_token": self.private_obs,
                   "obs_player2": th.from_numpy(self.get_obs_space()["player2"].low).zero_()}
            return obs, reward, self.done, {}

    def reset(self):
        self.steps = 0
        obs = self.env.reset()
        self.private_obs = th.from_numpy(self.private_obs_dist()) + 1 # +1 is important as 0 is no-op
        self.trajectory = {"actions": [],
                           "states": [obs],
                           "rewards": [],
                           "next_states": []}
        self.final_reward = None
        self.done = False
        return {"obs_player1": obs, "private_token": self.private_obs, "obs_player2": obs.clone().zero_()}

from PIL import Image
import torchvision.transforms as T
class MountainCar():

    def __init__(self, **kwargs):
        self.reset()
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step(action.item())
        if reward != 0.0:
            done = True
        return {"obs_player1": self.get_obs()}, reward, done, info

    def reset(self):

        self.env = gym.make('MountainCar-v0')

        self.env.reset()
        reward = 0.0
        ctr = 0.0
        while reward == 0.0: # Skip first pong game!
            _, reward, _, _ = self.env.step(0)
            ctr += 1
        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"obs_player1": self.get_obs()}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*3)}

    def get_obs(self):
        resize =T.Compose([
                        T.ToTensor(),
                        T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()
                    ])
        last_screen = resize(self.get_screen().copy())
        current_screen = resize(self.get_screen().copy())
        state = current_screen - last_screen
        return state

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        self.env.render()
        pass

    # Borrowed from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#input-extraction
    def get_car_location(self, env, screen_width):
        xmin = env.env.min_position
        xmax = env.env.max_position
        world_width = xmax - xmin
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CAR

    # Borrowed from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#input-extraction
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array')
        # Cart is in the lower half, so strip off the top and bottom of the screen
        screen_height, screen_width, _ = screen.shape
        # screen = screen[int(screen_height * 0.8), :]
        view_width = int(screen_width)
        car_location = self.get_car_location(self, screen_width)
        if car_location < view_width // 2:
            slice_range = slice(view_width)
        elif car_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(car_location - view_width // 2,
                                car_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, slice_range, :]
        return screen

class Simple4WayGrid():
    def __init__(self, max_steps, grid_dim=(4,4), step_penalty = 0.01):
        """
        Return only every ``skip``-th frame (frameskipping)

        :param env: the environment
        :param skip: number of ``skip``-th frame
        """
        self.grid_dim = grid_dim
        self.max_steps = max_steps #self.grid_dim[0] + self.grid_dim[1]
        self.step_penalty = step_penalty
        pass

    def reset(self):
        self.grid = th.zeros(self.grid_dim)
        self.agent_pos = [0,0]
        self.steps = 0
        self.done = False
        return {"obs_player1": self._get_state()}

    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        reward = 0.0
        info = {}

        assert not self.done, "TRUE!"
        assert action in list(range(4)), "Invalid action: {}".format(action)

        self.grid[self.agent_pos[0], self.agent_pos[1]] = 0.0
        if action == 0:
            self.agent_pos[0] = min(self.grid.shape[0]-1, self.agent_pos[0]+1)
        elif action == 1:
            self.agent_pos[1] = min(self.grid.shape[1]-1, self.agent_pos[1]+1)
        elif action == 2:
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        elif action == 3:
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)

        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1.0
        obs = self.grid.unsqueeze(0)
        self.steps += 1

        if self.agent_pos == [self.grid.shape[0]-1, self.grid.shape[1]-1]:
            reward = 1.0 - max(0.0, self.step_penalty * (self.steps - sum(self.grid.shape)))
            self.done = True
        elif self.steps == self.max_steps:
            reward = -1.0

        if self.steps == self.max_steps:
            self.done = True

        return {"obs_player1": obs}, reward, self.done, info

    def get_obs_space(self):
        return {"player1": gym.spaces.Box(self.grid.clone().zero_().numpy(), self.grid.clone().zero_().numpy()+1)}

    def get_action_space(self):
        return {"player1": gym.spaces.Discrete(4)}

    def get_available_actions(self):
        is_avail__action_0 = self.agent_pos[0] < self.grid.shape[0]-1
        is_avail__action_1 = self.agent_pos[1] < self.grid.shape[1]-1
        is_avail__action_2 = self.agent_pos[0] > 0
        is_avail__action_3 = self.agent_pos[1] > 0
        return {"player1": th.FloatTensor([is_avail__action_0,
                               is_avail__action_1,
                               is_avail__action_2,
                               is_avail__action_3]).bool()}

    def _get_state(self):
        g = th.zeros_like(self.grid)
        g[self.agent_pos[0], self.agent_pos[1]] = 1.0
        return g.unsqueeze(0) # add channel dim


class Simple4WayGridPrivateObs(Simple4WayGrid):

    def __init__(self, private_obs_dist, token_reward_factor, token_n, private_obs_dim, **kwargs):
        """
        Return only every ``skip``-th frame (frameskipping)

        :param env: the environment
        :param skip: number of ``skip``-th frame
        """
        super().__init__(**kwargs)

        assert callable(private_obs_dist), "Private Obs Dist must be a callable function to sample from!"
        self.private_obs_dist = private_obs_dist
        self.token_reward_factor = token_reward_factor
        self.private_obs_dim = private_obs_dim
        self.token_n = token_n
        pass

    def reset(self):
        obs = super().reset()["obs_player1"]
        self.private_obs = th.from_numpy(self.private_obs_dist()) # +1 is important as 0 is no-op
        self.trajectory = {"actions": [],
                           "states": [obs],
                           "rewards": [],
                           "next_states": [],
                           "avail_actions": [self.get_available_actions()["player1"]],
                           "next_avail_actions": []}
        self.final_reward = None
        self.done = False
        return {"obs_player1": obs, "private_token": self.private_obs, "obs_player2": obs.clone().zero_()}

    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        assert action != -1, "You should never have to call step on no-op in this environment!"

        if not self.done:
            action = action - 1  # because now 0 is no-op
            if self.steps > 0:
                self.trajectory["states"].append(self.trajectory["next_states"][-1])
                self.trajectory["avail_actions"].append(self.trajectory["next_avail_actions"][-1])
            obs, reward, done, info = super().step(action)
            obs = obs["obs_player1"]
            self.trajectory["actions"].append(action.clone()+1) # as 0 is no=op!
            self.trajectory["next_states"].append(obs.clone())
            self.trajectory["rewards"].append(reward)
            self.trajectory["next_avail_actions"].append(self.get_available_actions()["player1"])
            obs = {"obs_player1": obs, "private_token": self.private_obs, "obs_player2": obs.clone().zero_()}
            self.final_reward = reward
            self.done = done
            return obs, reward, self.done, info
        else:

            reward_p2 = 1.0 if action.to(self.private_obs.device) == self.private_obs else -1.0
            reward = self.final_reward * (1-self.token_reward_factor) + self.token_reward_factor * reward_p2

            obsp1 = self._get_state().clone()
            obsp1[:, obsp1.shape[1]-1, obsp1.shape[2]-1] = -1 # indicate that we are in the token step!
            obs = {"obs_player1": obsp1,
                   "private_token": self.private_obs,
                   "obs_player2": self._get_state().clone().zero_()}
            return obs, reward, self.done, {}

    def get_obs_space(self):
        return {"player1": gym.spaces.Box(self.grid.clone().zero_().numpy(), self.grid.clone().zero_().numpy()+1),
                "player2": gym.spaces.Box(th.zeros(self.grid.view(-1).shape[0] + 1).numpy(), th.zeros(self.grid.view(-1).shape[0] + 1).numpy() + 1)}

    def get_action_space(self):
        return {"player1": gym.spaces.Discrete(5), # includes no-op!
                "player2": gym.spaces.Discrete(self.token_n+1) # includes no-op!
                }

    def get_available_actions(self):
        if self.done:
            return {"player1": th.FloatTensor([1.0,
                                               0.0,
                                               0.0,
                                               0.0,
                                               0.0]).bool(),
                    "player2": th.FloatTensor([0.0]+[1.0]*self.token_n).bool()}
        else:
            is_avail__action_0 = self.agent_pos[0] < self.grid.shape[0]-1
            is_avail__action_1 = self.agent_pos[1] < self.grid.shape[1]-1
            is_avail__action_2 = self.agent_pos[0] > 0
            is_avail__action_3 = self.agent_pos[1] > 0
            return {"player1": th.FloatTensor([0.0,
                                               is_avail__action_0,
                                               is_avail__action_1,
                                               is_avail__action_2,
                                               is_avail__action_3]).bool(),
                    "player2": th.FloatTensor([1.0]+[0.0]*self.token_n).bool()}

class DummyVecEnv():

    def __init__(self, env_fn, batch_size):
        self.batch_size = batch_size
        self.envs = [env_fn() for _ in range(batch_size)]
        pass

    def step(self, actions, a=None):
        if a is None:
            obs_lst = []
            reward_lst = []
            done_lst = []
            info_lst = []
            for i in range(self.batch_size):
                obs, reward, done, info = self.envs[i].step(actions[i])
                obs_lst.append(obs)
                reward_lst.append(reward)
                done_lst.append(done)
                info_lst.append(info)

            return th.stack(obs_lst) if not isinstance(obs_lst[0], dict) else {k:th.stack([o[k] for o in obs_lst]) for k,v in obs_lst[0].items()}, \
                   th.FloatTensor(reward_lst), \
                   done_lst, \
                   info_lst
        else:
            obs, reward, done, info = self.envs[a].step(actions)
            return obs, th.FloatTensor([reward]), done, info

    def reset(self, idx=None):
        if idx is None:
            obs_lst = []
            for i in range(self.batch_size):
                obs = self.envs[i].reset()
                obs_lst.append(obs)
            return th.stack(obs_lst) if not isinstance(obs_lst[0], dict) else {k:th.stack([o[k] for o in obs_lst]) for k,v in obs_lst[0].items()}
        else:
            return self.envs[idx].reset()

    def render(self, **kwargs):
        img = self.envs[0].render('rgb_array')
        return img

    def close(self):
        for i in range(self.batch_size):
            self.envs[i].close()

    def get_obs_space(self):
        return self.envs[0].get_obs_space()

    def get_action_space(self):
        return self.envs[0].get_action_space()

    def get_available_actions(self):
        avail_actions_dct = defaultdict(lambda: [])
        for i in range(self.batch_size):
            avail_actions = self.envs[i].get_available_actions()
            for k, v in avail_actions.items():
                avail_actions_dct[k].append(v)
        return {k:th.stack(v) for k, v in avail_actions_dct.items()}

    def get_trajectory(self, idx=None):
        if idx is not None:
            return self.envs[idx].trajectory
        else:
            return [e.trajectory for e in self.envs]
    pass


class ConvNet(nn.Module):
    def __init__(self, n_actions, input_shape, use_maxpool=False):
        super(ConvNet, self).__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape

        if not use_maxpool:
            self.layer1 = nn.Sequential(
                nn.Conv2d(self.input_shape[0], 16, kernel_size=(2, 2), stride=1, padding=1),
                nn.ReLU())
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(self.input_shape[0], 16, kernel_size=(2, 2), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(), )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(), )
        self.drop_out = nn.Dropout(0.2)

        out_dims = self._get_fc_dims()
        self.fc1 = nn.Sequential(nn.Linear(out_dims[1] * out_dims[2] * out_dims[3], 64),
                                 nn.ReLU())
        self.fc2 = nn.Linear(64, self.n_actions)

    def _get_fc_dims(self):
        dummy = th.zeros((1, *self.input_shape), device=next(self.parameters()).device)
        y = self.layer1(dummy)
        y = self.layer2(y)
        y = self.layer3(y)
        return y.shape

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.fc1(y.view(y.shape[0], -1))
        y = self.fc2(y)
        return y


class ConvNetMini(nn.Module):
    def __init__(self, n_actions, input_shape, use_maxpool=False):
        super(ConvNetMini, self).__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 16, kernel_size=(3, 3), stride=1, padding=3),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(), )

        out_dims = self._get_fc_dims()
        self.fc1 = nn.Sequential(nn.Linear(out_dims[1] * out_dims[2] * out_dims[3], 64),
                                 nn.ReLU())
        self.fc2 = nn.Linear(64, self.n_actions)

    def _get_fc_dims(self):
        dummy = th.zeros((1, *self.input_shape), device=next(self.parameters()).device)
        y = self.layer1(dummy)
        y = self.layer2(y)
        return y.shape

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.fc1(y.view(y.shape[0], -1))
        y = self.fc2(y)
        return y

class MLPNetDeep(nn.Module):
    def __init__(self, n_actions, input_shape, other_shape_dct=None):
        super(MLPNetDeep, self).__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.other_shape_dct = other_shape_dct
        self.other_shape_lin = sum([reduce(operator.mul, v) for v in other_shape_dct.values()]) if other_shape_dct != None else 0

        self.fc0 = nn.Sequential(nn.Linear(reduce(operator.mul, self.input_shape) + self.other_shape_lin, 64),
                                 nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(64, 64),
                                 nn.ReLU())
        self.fc2 = nn.Linear(64, self.n_actions)

    def forward(self, x, other_dct=None, is_sequence=False, **kwargs):
        if not is_sequence:
            z = x.view(x.shape[0],-1)
            if other_dct is not None:
                z = th.cat(
                    [z] + [other_dct[k].to(x.device).view(other_dct[k].shape[0], -1) for k in sorted(list(other_dct.keys()))], -1)
        else:
            z = x.view(x.shape[0]*x.shape[1], -1)
            if other_dct is not None:
                z = th.cat(
                    [z] + [other_dct[k].to(x.device).view(other_dct[k].shape[0]*other_dct[k].shape[1], -1) for k in sorted(list(other_dct.keys()))], -1)
        y = self.fc0(z)
        y = self.fc1(y)
        y = self.fc2(y)
        return {"out": y if not is_sequence else y.view(x.shape[0], x.shape[1], -1)}

class MLPNet(nn.Module):
    def __init__(self, n_actions, input_shape, other_shape_dct=None):
        super(MLPNet, self).__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.other_shape_dct = other_shape_dct
        self.other_shape_lin = sum([reduce(operator.mul, v) for v in other_shape_dct.values()]) if other_shape_dct != None else 0


        self.fc1 = nn.Sequential(nn.Linear(64, 64),
                                 nn.ReLU())
        self.fc2 = nn.Linear(64, self.n_actions)

    def forward(self, x, other_dct=None):
        x = x.view(x.shape[0],-1)
        if other_dct is not None:
            x = th.cat([x]+[other_dct[k].to(x.device).view(x.shape[0],-1) for k in sorted(list(other_dct.keys()))], -1)
        y = self.fc1(x)
        y = self.fc2(y)
        return y

class MLPLSTMNetDeep(nn.Module):
    def __init__(self, n_actions, input_shape, other_shape_dct=None):
        super(MLPLSTMNetDeep, self).__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.other_shape_dct = other_shape_dct
        self.rnn_hidden_dim = 64
        self.other_shape_lin = sum([reduce(operator.mul, v) for v in other_shape_dct.values()]) if other_shape_dct != None else 0
        self.fc0 = nn.Sequential(nn.Linear(reduce(operator.mul, self.input_shape) + self.other_shape_lin, self.rnn_hidden_dim),
                                 nn.ReLU())

        self.rnn = nn.LSTM(self.rnn_hidden_dim, self.rnn_hidden_dim, 1, batch_first=False)
        self.fc1 = nn.Sequential(nn.Linear(self.rnn_hidden_dim, 64),
                                 nn.ReLU())
        self.fc2 = nn.Linear(64, self.n_actions)

    def forward(self, x, other_dct=None, is_sequence=False, hidden=None):
        if is_sequence:
            z = x.view(x.shape[0], x.shape[1], -1)
            if other_dct is not None:
                z = th.cat(
                    [z] + [other_dct[k].to(x.device).view(other_dct[k].shape[0], other_dct[k].shape[1], -1) for k in sorted(list(other_dct.keys()))], -1)
            hidden = th.zeros((x.shape[1], self.rnn_hidden_dim), device=z.device).unsqueeze(0)
            x = F.relu(self.fc0(z))
            lstm_out, hidden = self.rnn(x, (hidden, hidden.clone()))

            q = self.fc2(lstm_out)
            return {"out": q if is_sequence else q.squeeze(0), "hidden": hidden}
        else:
            z = x.view(x.shape[0], -1)
            if other_dct is not None:
                z = th.cat(
                    [z] + [other_dct[k].to(x.device).view(other_dct[k].shape[0], -1) for k in sorted(list(other_dct.keys()))], -1)
            z = z.unsqueeze(0)

            for s in z:
                x = F.relu(self.fc0(s))
                hidden = hidden.reshape(-1, self.rnn_hidden_dim)
                hidden = self.rnn(x, hidden)
                o = self.fc1(hidden)
                q = self.fc2(o)

            return {"out": q if is_sequence else q.squeeze(0), "hidden": hidden}


class TabularGridNet(nn.Module):
    def __init__(self, n_actions, grid_shape, max_steps):
        super(TabularGridNet, self).__init__()
        self.n_actions = n_actions
        self.grid_shape = grid_shape
        self.max_t = max_steps + 1

        self.params = nn.Parameter(th.zeros([self.max_t, *self.grid_shape, self.n_actions]).normal_(), requires_grad=True)
        pass

    @staticmethod
    def where(cond, x_1, x_2):
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def forward(self, x, other_dct=None, is_sequence=False):

        if is_sequence:
            xs = x.view(-1, 1, *x.shape[-2:])
            ts = other_dct["t"].long().view(-1)
        else:
            xs = x
            ts = other_dct["t"].long().view(-1)

        grid_x = th.argmax(xs.sum(dim=-2), dim=-1).squeeze()
        grid_y = th.argmax(xs.sum(dim=-1), dim=-1).squeeze()
        p = self.params[ts,0,grid_x, grid_y, :]

        if is_sequence:
            return {"out": p.view(x.shape[0], x.shape[1], *p.shape[1:])}
        else:
            return {"out": p}


class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim=32, hypernet_layers=2, hypernet_embed=64):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim

        self.embed_dim = mixing_embed_dim

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class MountainCarNet(nn.Module):
    # from: https://github.com/greatwallet/mountain-car/blob/master/dqn_models.py

    def __init__(self, h, w, outputs):
        super(MountainCarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, *args, **kwargs):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return {"out": self.head(x.view(x.size(0), -1))}


class PongNet(th.nn.Module):

    def __init__(self, obs_space, n_actions=6):
        super().__init__()
        from stable_baselines3_lib.torch_layers import NatureCNN
        self.net = NatureCNN(obs_space)
        self.linear_q = th.nn.Linear(512, n_actions)

    def forward(self, input, other_dct, is_sequence=None):
        shp0 = input.shape[0]
        shp1 = input.shape[1]
        input = input.view(-1, *input.shape[-3:])
        # stacked_input = th.cat([input, th.zeros_like(input)[..., :1]+other_dct["log_beta"]], -1)
        stacked_input = th.cat([input, other_dct["log_beta"].unsqueeze(1).unsqueeze(1).repeat(1,84,84,1)], -1)
        out = self.net(stacked_input.permute(0,3,1,2))
        out = self.linear_q(out)
        return {"out": out if not is_sequence else out.view(shp0, shp1, *input.shape[-3:])}


class Buffer():

    def __init__(self, buffer_scheme, buffer_len, buffer_device):
        self.buffer_scheme = buffer_scheme
        self.buffer_len = buffer_len
        self.buffer_device = buffer_device
        self.flush()
        pass

    def flush(self):
        self.buffer_content = {}
        for k, v in self.buffer_scheme.items():
            self.buffer_content[k] = []

    def insert(self, dct):

        for k in self.buffer_scheme.keys():
            v = dct[k]
            self.buffer_content[k].append(v.to(self.buffer_device))

        pass

    def sample(self, sample_size, mode="transitions"):

        if mode in ["transitions"]:
            sz = self.size(mode="transitions")
            sample_ids = np.random.randint(1, sz + 1, size=sample_size)
            cs = np.cumsum([0] + [v.shape[0] for v in self.buffer_content[list(self.buffer_content.keys())[0]]])
            main_idxs = np.searchsorted(cs, sample_ids) - 1
            sub_idxs = [sid - cs[mid] - 1 for mid, sid in zip(main_idxs, sample_ids)]
            ret_dict = {}
            for k in self.buffer_scheme.keys():
                ret_dict[k] = th.stack(
                    [self.buffer_content[k][mid][sid, ...] for mid, sid in zip(main_idxs, sub_idxs)])

        elif mode in ["episodes"]:
            sz = self.size(mode="episodes")
            sample_ids = np.random.randint(0, sz, size=sample_size)
            ret_dict = {}
            for k in self.buffer_scheme.keys():
                ret_dict[k] = pad_sequence([self.buffer_content[k][sid] for sid in sample_ids])
                ret_dict[k + "__seq_mask"] = pad_sequence([self.buffer_content[k][sid].clone().zero_()+1 for sid in sample_ids])
                ret_dict[k + "__seq_len"] = th.LongTensor([self.buffer_content[k][sid].shape[0] for sid in sample_ids])
        else:
            assert False, "Sampling mode '{}' unknown!".format(mode)
        # return {"current": ret_dict, "next": ret_dict_next} # sampling a full transition
        return ret_dict

    def size(self, mode="episodes"):
        if mode == "episodes":
            return len(self.buffer_content[list(self.buffer_content.keys())[0]])
        elif mode == "transitions":
            return sum([v.shape[0] for v in self.buffer_content[list(self.buffer_content.keys())[0]]])
        else:
            raise Exception("Unknown size mode: {}".format(mode))