from collections import defaultdict
import gym
import numpy as np
import torch as th
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

#from stable_baselines3_lib import pong
from sb3.common.env_util import make_atari_env
from sb3.common.vec_env import VecFrameStack

GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]

class PongN():

    def __init__(self, **kwargs):
        self.pong_n_games = kwargs.get("pong_n_games", 21)
        self.reset()
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        if reward != 0.0:
            self.game_ctr += 1
        if self.game_ctr == self.pong_n_games:
            done = 1
        else:
            done = 0
        return {"player1": th.from_numpy(obs[0])}, reward[0], done, info

    def reset(self):
        self.env = make_atari_env('PongNoFrameskip-v4', n_envs=1,
                                       seed=0, wrapper_kwargs={"noop_max":0})  # wrapper_kwargs={"clip_reward": False, "pong_just_one_step": True},
        self.env = VecFrameStack(self.env, n_stack=4)
        obs = self.env.reset()
        self.game_ctr = 0
        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"player1": th.from_numpy(obs[0])}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*6)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        return self.env.render()

class PongNRestrictedActionSpace():

    def __init__(self, **kwargs):
        self.pong_n_games = kwargs.get("pong_n_games", 21)
        self.reset()
        pass

    def step(self, action):
        if action == 0:
            action_trans = 2
        elif action == 1:
            action_trans = 3
        else:
            raise Exception("Unknown action: {}".format(action))
        obs, reward, done, info = self.env.step([action_trans])
        if reward != 0.0:
            self.game_ctr += 1
        if self.game_ctr == self.pong_n_games:
            done = 1
        else:
            done = 0
        return {"player1": th.from_numpy(obs[0])}, reward[0], done, info

    def reset(self):
        self.env = make_atari_env('PongNoFrameskip-v4', n_envs=1,
                                       seed=0, wrapper_kwargs={"noop_max":0})  # wrapper_kwargs={"clip_reward": False, "pong_just_one_step": True},
        self.env = VecFrameStack(self.env, n_stack=4)
        obs = self.env.reset()
        self.game_ctr = 0
        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"player1": th.from_numpy(obs[0])}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*2)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        return self.env.render()


class Breakout():

    def __init__(self, **kwargs):
        self.reset()
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        if reward != 0.0:
            self.game_ctr += 1
        return {"player1": th.from_numpy(obs[0])}, reward[0], done, info

    def reset(self):
        self.env = pong.make_atari_env('BreakoutNoFrameskip-v4', n_envs=1,
                                       seed=0, wrapper_kwargs={"noop_max":0})  # wrapper_kwargs={"clip_reward": False, "pong_just_one_step": True},
        self.env = pong.VecFrameStack(self.env, n_stack=4)
        obs = self.env.reset()
        self.game_ctr = 0
        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"player1": th.from_numpy(obs[0])}

    def get_available_actions(self):
        return  {"player1": th.BoolTensor([True]*4)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        return self.env.render()

class BreakoutSB3():

    def __init__(self, **kwargs):
        # self.pong_n_games = kwargs.get("pong_n_games", 21)
        from rl_baselines3_zoo_lib import train
        expmanager = train.main(["--algo", "dqn", "--env",
                                 "BreakoutNoFrameskip-v4",
                                 "-i",
                                 "models/BreakoutNoFrameskip-v4/BreakoutNoFrameskip-v4.zip",
                                 "-n", "5000"])
        _ = expmanager.setup_experiment()
        self.env = expmanager.env
        self.env.venv.venv.envs[0] = self.env.venv.venv.envs[0].env.env # get ClipRewardEnv out!
        self.env.reset()
        pass

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        if reward != 0.0:
            self.game_ctr += 1
        return {"player1": th.from_numpy(obs[0])}, reward[0], done, info

    def reset(self):

        obs = self.env.reset()
        self.game_ctr = 0
        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"player1": th.from_numpy(obs[0])}

    def get_available_actions(self):
        return {"player1": th.BoolTensor([True]*4)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        return self.env.render()

class CartpoleSB3():

    def __init__(self, **kwargs):
        # self.pong_n_games = kwargs.get("pong_n_games", 21)
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

        return {"player1": th.from_numpy(obs[0])}, reward[0], done, info

    def reset(self):

        obs = self.env.reset()
        self.game_ctr = 0
        self.steps = 0
        self.final_reward = None
        self.done = False
        return {"player1": th.from_numpy(obs[0])}

    def get_available_actions(self):
        return {"player1": th.BoolTensor([True]*self.env.action_space.n)}

    def get_obs_space(self):
        return {"player1": self.env.observation_space}

    def get_action_space(self):
        return {"player1": self.env.action_space}

    def render(self, *args, **kwargs):
        return self.env.render()


class Simple4WayGrid():
    def __init__(self, max_steps, grid_dim=(4,4), step_penalty = 0.01):
        """
        Return only every ``skip``-th frame (frameskipping)

        :param env: the environment
        :param skip: number of ``skip``-th frame
        """
        self.grid_dim = grid_dim
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        pass

    def reset(self):
        self.grid = th.zeros(self.grid_dim)
        self.agent_pos = [0,0]
        self.steps = 0
        self.done = False
        return {"player1": self._get_state()}

    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        reward = 0.0
        info = [{}]

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
            reward = 1.0
            self.done = True
        elif self.steps == self.max_steps:
            reward = 0.0
            #self.done = True
        if self.steps == self.max_steps:
            self.done = True

        return {"player1": obs}, reward, self.done, info

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

            reward_p2 = 1.0 if action.to(self.private_obs.device) == self.private_obs else 0.0
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
        self.envs = [env_fn.x() if isinstance(env_fn, CloudpickleWrapper) else env_fn() for _ in range(batch_size)]
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
                info_lst.append(info[0])

            return th.stack(obs_lst) if not isinstance(obs_lst[0], dict) else {k:th.stack([o[k] for o in obs_lst]) for k,v in obs_lst[0].items()}, \
                   th.FloatTensor(reward_lst), \
                   done_lst, \
                   info_lst
        else:
            obs, reward, done, info = self.envs[a].step(actions)
            return obs, th.FloatTensor([reward]), done, info

    def reset(self, idx=None):
        if idx is None:
            idx_lst = list(range(self.batch_size))
        else:
            idx_lst = [idx] if isinstance(idx, int) else idx
        obs_lst = []
        for i in idx_lst:
            obs = self.envs[i].reset()
            obs_lst.append(obs)
        if len(idx_lst) == 1:
            return obs_lst[0] if not isinstance(obs_lst[0], dict) else {k: v for  k, v in obs_lst[0].items()}
        else:
            return th.stack(obs_lst) if not isinstance(obs_lst[0], dict) else {k:th.stack([o[k] for o in obs_lst]) for k,v in obs_lst[0].items()}

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

    def get_available_actions(self, idx=None):
        if idx is None:
            idx_lst = list(range(self.batch_size))
        else:
            idx_lst = [idx] if isinstance(idx, int) else idx

        avail_actions_dct = defaultdict(lambda: [])
        for i in idx_lst:
            avail_actions = self.envs[i].get_available_actions()
            for k, v in avail_actions.items():
                avail_actions_dct[k].append(v)
        if len(idx_lst) == 1:
            return {k:v[0] for k, v in avail_actions_dct.items()}
        else:
            return {k:th.stack(v) for k, v in avail_actions_dct.items()}

    def get_trajectory(self, idx=None):
        if idx is None:
            idx_lst = list(range(self.batch_size))
        else:
            idx_lst = [idx] if isinstance(idx, int) else idx
        return [self.envs[_id].trajectory for _id in idx_lst]
    pass

import multiprocessing as mp
from vec_env import CloudpickleWrapper, clear_mpi_env_vars

def dummy_vec_env_worker(remote, parent_remote, env_fn_wrappers, batch_size=None):
    vec_env = DummyVecEnv(env_fn=env_fn_wrappers, batch_size=batch_size)
    parent_remote.close()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(vec_env.step(data))
            elif cmd == 'reset':
                remote.send(vec_env.reset(data))
            elif cmd == 'render':
                remote.send(vec_env.render())
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((vec_env.envs[0].get_obs_space(),
                                                vec_env.envs[0].get_action_space())))
            elif cmd == 'get_available_actions':
                remote.send(vec_env.get_available_actions(data))
            elif cmd == 'get_trajectory':
                remote.send(vec_env.get_trajectory(data))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        vec_env.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])
    return [l__ for l_ in l for l__ in l_]

def split_arr(arr, spltarr):
    split_actions = []
    ctr = 0
    for bs in spltarr:
        chunk = arr[ctr:ctr + bs]
        split_actions.append(chunk)
        ctr += bs
    return split_actions

class ParallelEnv():
    # heavily inspired by: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py

    def __init__(self, env_fn, batch_sizes=None, context='spawn', in_series=1):
        self.batch_sizes = batch_sizes
        self.waiting = False
        self.closed = False
        self.nremotes = len(batch_sizes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=dummy_vec_env_worker, args=(work_remote, remote,
                                                                  CloudpickleWrapper(env_fn), batch_size))
                   for (work_remote, remote, batch_size) in zip(self.work_remotes, self.remotes, batch_sizes)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        self.observation_space, self.action_space = self.remotes[0].recv().x
        self.viewer = None

        self.idx_lst = []
        ctr = 0
        for bs in batch_sizes:
            self.idx_lst.append(list(range(ctr, ctr+bs)))
            ctr += bs
        pass

    def idx2workeridx(self, _idx):
        for i, idxl in enumerate(self.idx_lst):
            if idxl[0] <= _idx and _idx <= idxl[-1]:
                return i, idxl.index(_idx)
        return -1

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        # split actions into appropriate pieces
        split_actions = split_arr(actions, self.batch_sizes)
        self._assert_not_closed()
        for remote, action in zip(self.remotes, split_actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        obses = {k: th.cat([r[0][k] for r in results], 0) for k in ["player1"]}
        rews = th.cat([r[1] for r in results], 0)
        dones = [x for r in results for x in r[2]]
        infos = [x for r in results for x in r[3]]
        return obses, rews, dones, infos

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def reset(self, idx=None):
        if idx is not None:
            # TODO IDX SPLITTING!
            # assign each idx to the proper worker subid
            if isinstance(idx, int):
                idx = [idx]
            dct = {}
            for _idx in idx:
                c, i = self.idx2workeridx(_idx)
                if not c in dct:
                    dct[c] = []
                dct[c].append(i)

            self._assert_not_closed()
            for c in sorted(dct.keys()):
                self.remotes[c].send(('reset', dct[c]))
            obs_lst = [self.remotes[c].recv() for c in sorted(dct.keys())]
            if len(dct.keys()) == 1:
                return obs_lst[0] if not isinstance(obs_lst[0], dict) else {k: v for k, v in obs_lst[0].items()}
            else:
                return th.stack(obs_lst) if not isinstance(obs_lst[0], dict) else {k: th.stack([o[k] for o in obs_lst])
                                                                                   for k, v in obs_lst[0].items()}
        else:
            for pipe in self.remotes:
                pipe.send(('reset', None))
            obs_lst = [pipe.recv() for pipe in self.remotes]
            return th.cat(obs_lst, 0) if not isinstance(obs_lst[0], dict) else {k:th.cat([o[k] for o in obs_lst], 0) for k,v in obs_lst[0].items()}

    def close(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_obs_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def get_available_actions(self, idx=None):
        if idx is not None:
            # assign each idx to the proper worker subid
            if isinstance(idx, int):
                idx = [idx]
            dct = {}
            for _idx in idx:
                c, i = self.idx2workeridx(_idx)
                if not c in dct:
                    dct[c] = []
                dct[c].append(i)

            self._assert_not_closed()
            for c in sorted(dct.keys()):
                self.remotes[c].send(('get_available_actions', dct[c]))
            avail_actions_lst = [self.remotes[c].recv() for c in sorted(dct.keys())]
            if len(dct.keys()) == 1:
                return avail_actions_lst[0] if not isinstance(avail_actions_lst[0], dict) else {k: v for k, v in avail_actions_lst[0].items()}
            else:
                return th.stack(avail_actions_lst) if not isinstance(avail_actions_lst[0], dict) else {k: th.stack([o[k] for o in avail_actions_lst])
                                                                                   for k, v in avail_actions_lst[0].items()}
        else:
            for pipe in self.remotes:
                pipe.send(('get_available_actions', None))
            avail_actions_lst = [pipe.recv() for pipe in self.remotes]
            return th.stack(avail_actions_lst) if not isinstance(avail_actions_lst[0], dict) \
                else {k:th.stack([o[k] for o in avail_actions_lst]) for k,v in avail_actions_lst[0].items()}

    def get_trajectory(self, idx=None):
        if idx is not None:
            # assign each idx to the proper worker subid
            if isinstance(idx, int):
                idx = [idx]
            dct = {}
            for _idx in idx:
                c, i = self.idx2workeridx(_idx)
                if not c in dct:
                    dct[c] = []
                dct[c].append(i)

            self._assert_not_closed()
            for c in sorted(dct.keys()):
                self.remotes[c].send(('get_trajectory', dct[c]))
            trajectory_lst = [self.remotes[c].recv() for c in sorted(dct.keys())]
            return th.stack(trajectory_lst) if not isinstance(trajectory_lst[0], dict) \
                else {k:th.stack([o[k] for o in trajectory_lst]) for k,v in trajectory_lst[0].items()}
        else:
            for pipe in self.remotes:
                pipe.send(('get_trajectory', None))
            trajectory_lst = [pipe.recv() for pipe in self.remotes]
            return th.stack(trajectory_lst) if not isinstance(trajectory_lst[0], dict) \
                else {k:th.stack([o[k] for o in trajectory_lst]) for k,v in trajectory_lst[0].items()}
    pass


def get_make_env_fn(env_tag, env_args):

    # Create environment
    env_registry = {"simple_4way_grid": Simple4WayGrid, "mini_grid": None, "pong": PongN, "breakout": Breakout, "breakoutsb3": BreakoutSB3,
                    "pongres": PongNRestrictedActionSpace, "cartpolesb3": CartpoleSB3}
    assert env_tag in env_registry, "Environment {} not in registry!".format(env_tag)

    make_env_fn = None
    if env_tag in ["simple_grid", "simple_4way_grid"]:
        def _make_env_fn():
            env = env_registry[env_tag](max_steps=env_args.env_arg_max_steps, grid_dim=env_args.env_arg_grid_dim)
            env.max_steps = env_args.env_arg_max_steps  # min(env.max_steps, max_steps)
            return env

    elif env_tag in ["pong"]:
        def _make_env_fn():
            env = env_registry[env_tag]()
            return env

    elif env_tag in ["breakout"]:
        def _make_env_fn():
            env = env_registry[env_tag]()
            return env

    elif env_tag in ["breakoutsb3"]:
        """
        from rl_baselines3_zoo_lib import train
        expmanager = train.main(["--algo", "dqn", "--env",
                                 "BreakoutNoFrameskip-v4",
                                 "-i",
                                 "models/BreakoutNoFrameskip-v4/BreakoutNoFrameskip-v4.zip",
                                 "-n", "5000"])
        model = expmanager.setup_experiment()
        """
        def _make_env_fn():
            env = env_registry[env_tag]()
            return env

    elif env_tag in ["cartpolesb3"]:
        def _make_env_fn():
            env = env_registry[env_tag]()
            return env

    elif env_tag in ["pongres"]:
        def _make_env_fn():
            env = env_registry[env_tag]()
            return env

    return _make_env_fn
