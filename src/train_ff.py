import argparse
from collections import defaultdict
from functools import partial, reduce
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
from gym_minigrid.wrappers import * # Test importing wrappers
import math
import matplotlib.pyplot as plt
import numpy as np
import pprint
import torch as th
from torch import nn
from torch import optim, nn
import pickle, json

from buffer import Buffer
from envs import get_make_env_fn, DummyVecEnv, ParallelEnv
from models import get_models
from utils import masked_softmax

########################################################################################################################
# TESTED CONFIGURATIONS
########################################################################################################################

"""
--exp-name "final_3x3_" --net "mlp_deep" --net-device "cuda:0" --log-betas-range-rollout 1 10 \
    --log-betas-eval 1 2 3 5 7 10 --env-arg-grid-dim 3 3 --env-arg-max-steps 6 --buffer-sampling-mode episodes --lr 0.001
    
--exp-name "final_4x4_" --net "mlp_deep" --net-device "cuda:0" --log-betas-range-rollout 0.1 7.5 --log-betas-eval 0.1 1 2 3 5 7 \
    --env-arg-grid-dim 4 4 --env-arg-max-steps 8 --buffer-sampling-mode episodes --lr 0.001
"""

########################################################################################################################
# ARGPARSE -- START
########################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--algo", default="soft_q", type=str, help="learning algorithm")
parser.add_argument("--algo-variant", default="double_q", type=str, help="learning algorithm variant")
parser.add_argument("--bs_rollout", default=32, type=int, help="batch size used during rollout")
parser.add_argument("--bs_learn", default=64, type=int, help="batch size used during learning")
parser.add_argument("--buffer-device", default="cuda:0", type=str, help="device buffer is stored on")
parser.add_argument("--buffer-max-n-episodes", default=1000, type=int, help="length of the device buffer")
parser.add_argument("--net", default="mlp", type=str, help="device models are stored on")
parser.add_argument("--net-device", default="cuda:0", type=str, help="device models are stored on")
parser.add_argument("--env", default="simple_4way_grid", type=str, help="environment string")
parser.add_argument("--env-arg-scenario-name", default="MiniGrid-Empty-5x5-v0", type=str, help="environment scenario name")
parser.add_argument("--env-arg-max-steps", default=20, type=int, help="environment max steps")
parser.add_argument("--env-arg-grid-dim", default=[4,4], nargs=2, type=int, help="grid dimensions tuple(x,y) of grid env")
parser.add_argument("--epsilon-final", default=0.05, type=float, help="epsilon schedule final")
parser.add_argument("--epsilon-timescale", default=10000, type=float, help="epsilon schedule timescale")
parser.add_argument("--exp-name", required=True, default="unnamed", type=str, help="learning algorithm")
parser.add_argument("--eval-every-x-episodes", default=3000, type=int, help="eval every x episodes")
parser.add_argument("--gamma", default=0.99, type=float, help="gamma [0,1] (discount parameter)")
parser.add_argument("--learn-every-x-episodes", default=1, type=int, help="learn every x episodes")
parser.add_argument("--learn-every-x-steps", default=-1, type=int, help="learn every x episodes")
parser.add_argument("--learn-n-times", default=10, type=int, help="learn x times per learn update")
parser.add_argument("--log-betas-range-rollout", default=(0.01, 7.0), type=float, nargs=2,  help="range of log betas to be sampled during rollout (soft_q only)")
parser.add_argument("--log-betas-eval", default=[0.01, 1, 1.5] + list(range(2, 7)), type=float, nargs="+", help="range of log betas to be sampled during rollout")
parser.add_argument("--lr", default=1E-4, type=float, help="learning rate")
parser.add_argument("--n-episodes-rollout-total", default=1000000, type=int, help="number of episodes to rollout in total")
parser.add_argument("--n-episode-rollout-per-outer-loop", default=100, type=int, help="number of episodes to rollout in total")
parser.add_argument("--n-episodes-eval", default=50, type=int, help="number of episodes to evaluate on")
parser.add_argument("--n-vec-envs", default=1, type=int, help="number of vec envs to run")
parser.add_argument("--optimizer", default="adam", help="name of optimizer to use")
parser.add_argument("--update-target-net-every-x-episodes", default=10, help="update target net every x episodes")
parser.add_argument("--input-t", default=True, type=bool, help="update target net every x episodes")
parser.add_argument("--pretrain-weights", default=None, type=str, help="point to .pt file with pretrain weights")

def get_free_device_memory(device_id):
    r = th.cuda.memory_reserved(device_id)
    a = th.cuda.memory_allocated(device_id)
    f = r - a
    return f

########################################################################################################################
# ARGPARSE -- STOP
########################################################################################################################

if __name__ == '__main__':

    best_score = -10000

    import uuid
    exphash = uuid.uuid4().hex[:6]

    args = parser.parse_args()
    pprint.pprint(vars(args))

    # get make env fn
    make_env_fn = get_make_env_fn(args.env, args)

    # create vec env
    if args.n_vec_envs == 1:
        vec_env = DummyVecEnv(env_fn=make_env_fn, batch_size=args.bs_rollout)
        eval_vec_env = DummyVecEnv(env_fn=make_env_fn, batch_size=1)
    else:
        vec_env = ParallelEnv(env_fn=make_env_fn,
                              batch_sizes=[len(p) for p in np.array_split(list(range(args.bs_rollout)), args.n_vec_envs)])
        eval_vec_env = ParallelEnv(env_fn=make_env_fn,
                                   batch_sizes=[1])

    # Create other_shapes_dct
    other_shapes_dct = {}
    if args.algo == "soft_q":
        other_shapes_dct["log_beta"] = (1,)
    if args.input_t:
        other_shapes_dct["t"] = (1,)

    # create models
    models, models_get_params = get_models(net_tag=args.net,
                                           n_actions=vec_env.get_action_space()["player1"].n,
                                           max_steps=args.env_arg_max_steps,
                                           obs_shape=vec_env.reset()["player1"][0].shape,
                                           other_shapes_dct=other_shapes_dct,
                                           device=args.net_device)

    if args.pretrain_weights is not None:
        models["net"] = th.load(args.pretrain_weights)
        models["target_net"] = th.load(args.pretrain_weights)

    # Create network optimizer
    assert args.optimizer in ["adam"], "Optimizer {} not in registry!".format(args.optimizer)
    optimizer = optim.Adam(models_get_params(), lr=args.lr)

    # Set up epsilon fn
    epsilon_fn = lambda ep_t: max(1.0 * math.exp(-ep_t / args.epsilon_timescale), args.epsilon_final)

    def tensor_helper(lst_dct, key, device):
        vals = [v[key][-1] for v in lst_dct]
        tens = th.stack(vals).to(device)
        return tens

    def stack_helper(dct, device):
        new_dct = {}
        for k, v in dct.items():
            if isinstance(v, (list, tuple)):
                new_dct[k] = th.stack(v).to(device)
            else:
                new_dct[k] = v
        return new_dct


    buffer_scheme = {"obs": vec_env.get_obs_space()["player1"].high.shape,
                     "action": (vec_env.get_action_space()["player1"].n,),
                     "avail_actions": (vec_env.get_action_space()["player1"].n,),
                     "reward": (1,),
                     "epsilon": (1,),
                     "log_beta": (1,),
                     }
    if args.input_t:
        buffer_scheme["t"] = (1,)
    buffer = Buffer(buffer_scheme, buffer_len=args.buffer_max_n_episodes, buffer_device=args.buffer_device)

    ro_n_episodes = 0
    lt_ro_n_episodes = 0
    lt_ro_n_steps = 0
    lt_ro_n_steps_print_ctr = 0
    le_ro_n_episodes = 0
    lu_ro_n_episodes = 0
    ro_n_steps = 0
    ro_lst_dct = [{"done": -1} for _ in range(args.bs_rollout)]

    from collections import deque
    rollout_stats = {"return": deque(maxlen=100), "epsilon": deque(maxlen=100)}
    while ro_n_episodes < args.n_episodes_rollout_total:

        ################################ ROLLOUTS START ################################################################
        for s in range(args.n_episode_rollout_per_outer_loop):
            # all the envs that need reset: reset
            for _idx in range(args.bs_rollout):
                if ro_lst_dct[_idx]["done"] in [1,-1]:
                    if ro_lst_dct[_idx]["done"] == 1:
                        buffer.insert(stack_helper(ro_lst_dct[_idx], device=args.net_device))
                        rollout_stats["return"].append(th.zeros(1, device=args.net_device) + sum(ro_lst_dct[_idx]["reward"]))
                        rollout_stats["epsilon"].append(th.mean(th.stack(ro_lst_dct[_idx]["epsilon"])))
                        ro_n_episodes += 1
                    log_beta = (args.log_betas_range_rollout[0] - args.log_betas_range_rollout[1]) * \
                               th.rand((1, ), device=args.net_device) + args.log_betas_range_rollout[1]
                    rst = vec_env.reset(_idx)["player1"]
                    avl = vec_env.get_available_actions(idx=_idx)["player1"].to(args.net_device)
                    ro_lst_dct[_idx] = {"done": 0,
                                        "obs": [rst],
                                        "action": [],
                                        "reward": [],
                                        "log_beta": [th.zeros(1, device=args.net_device) + log_beta],
                                        "avail_actions": [avl],
                                        "epsilon": [],
                                        }
                    if args.input_t:
                        ro_lst_dct[_idx]["t"] = [th.zeros(1, device=args.net_device)]

            # sample greedy actions
            other_dct = {"log_beta": tensor_helper(ro_lst_dct, "log_beta", args.net_device)}
            if args.input_t:
                other_dct["t"] = tensor_helper(ro_lst_dct, "t", args.net_device)
            q_values = models["net"](tensor_helper(ro_lst_dct, "obs", args.net_device), other_dct)["out"]
            ro_avail_actions = tensor_helper(ro_lst_dct, "avail_actions", args.net_device)

            policy = masked_softmax(q_values,
                                    mask=(ro_avail_actions == 0),
                                    beta=th.exp(tensor_helper(ro_lst_dct, "log_beta", args.net_device)),
                                    dim=-1)
            ro_action = th.multinomial(policy, 1)

            # add action exploration
            ro_epsilon = epsilon_fn(ro_n_episodes)
            ro_epsilon0 = ro_action.clone().float().uniform_()
            ro_action[ro_epsilon0 < ro_epsilon] = th.multinomial(ro_avail_actions.float(), 1)[ro_epsilon0 < ro_epsilon]
            obs, reward, done, info = vec_env.step(ro_action)
            ro_n_steps += args.bs_rollout

            # store step results
            for _idx in range(args.bs_rollout):
                ro_lst_dct[_idx]["action"].append(ro_action[_idx])
                ro_lst_dct[_idx]["obs"].append(obs["player1"][_idx])
                ro_lst_dct[_idx]["reward"].append(reward[_idx])
                ro_lst_dct[_idx]["done"]  = done[_idx]
                ro_lst_dct[_idx]["avail_actions"].append(vec_env.get_available_actions(idx=_idx)["player1"].to(args.net_device))
                ro_lst_dct[_idx]["epsilon"].append(th.zeros(1, device=args.net_device) + ro_epsilon)
                ro_lst_dct[_idx]["log_beta"].append(th.zeros(1, device=args.net_device) + log_beta)
                if args.input_t:
                    ro_lst_dct[_idx]["t"].append(ro_lst_dct[_idx]["t"][-1] + 1)
            pass

        if len(rollout_stats["return"]) == 100:

            print("{} episodes, {} steps:: mean train reward: {:.2f} @ mean epsilon: {:.2f}".format(
                ro_n_episodes,
                ro_n_steps,
                th.mean(th.stack(list(rollout_stats["return"]))).item(),
                th.mean(th.stack(list(rollout_stats["epsilon"]))).item()))
        ################################ ROLLOUTS STOP #################################################################


        ################################ TRAINING START ################################################################
        if ( (args.learn_every_x_steps == -1 and ro_n_episodes - lt_ro_n_episodes >= args.learn_every_x_episodes) or \
             (args.learn_every_x_steps != -1 and ro_n_steps - lt_ro_n_steps >= args.learn_every_x_steps) ) and \
                (buffer.size(mode="transitions") >= args.bs_learn):

            lt_ro_n_episodes = ro_n_episodes
            lt_ro_n_steps = ro_n_steps

            for _ in range(args.learn_n_times):

                samples = {k: v.to(args.net_device) for k, v in
                           buffer.sample(args.bs_learn, mode="transitions").items()}

                other_dct = {}
                if args.input_t:
                    other_dct["t"] = samples["t"]
                if args.algo == "soft_q":
                    other_dct["log_beta"] = samples["log_beta"]

                q_values = models["net"](samples["obs"], other_dct, is_sequence=False)["out"]
                q_value_taken = q_values.gather(-1, samples["action"].long())
                q_value_taken = q_value_taken

                if args.input_t:
                    other_dict_next = {"t": samples["t"]+1}
                if args.algo == "soft_q":
                    other_dict_next["log_beta"] = samples["log_beta"]
                target_q_values = models["target_net"](samples["next_obs"], other_dict_next, is_sequence=False)["out"]
                target_q_values[~samples["next_avail_actions"]] = -1E20
                target_q_value = (1. / th.exp(samples["log_beta"])) * th.logsumexp(
                    th.exp(samples["log_beta"]) * target_q_values,
                    dim=-1, keepdim=True)

                # the target_value of a terminal state is always 0!
                terminal_state_mask = samples["next_obs_is_terminal"].bool()
                target_q_value[terminal_state_mask] = 0.0

                target_value = args.gamma * target_q_value + samples["reward"].unsqueeze(-1) # reward is always in previous timestep

                loss = (target_value.detach() - q_value_taken) ** 2
                loss = loss.mean()

                if ro_n_steps - lt_ro_n_steps_print_ctr > 1000:
                    print("Loss (t_ep: {} steps: {}): {}".format(ro_n_episodes, ro_n_steps, loss.mean()))
                    lt_ro_n_steps_print_ctr = ro_n_steps

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if ro_n_episodes - lu_ro_n_episodes >= args.update_target_net_every_x_episodes:
            lu_ro_n_episodes = ro_n_episodes
            print("Updating target network...")
            models["target_net"].load_state_dict(models["net"].state_dict())
        ################################ TRAINING STOP #################################################################

        ################################ EVAL START ####################################################################
        if ro_n_episodes - le_ro_n_episodes >= args.eval_every_x_episodes:
            print("Free device memory (eval): {}".format(get_free_device_memory(0)))
            print("Saving model...")
            import os
            path = os.path.join("results", "{}__{}".format(args.exp_name, exphash))
            if not os.path.exists(path):
                os.makedirs(path)

            print("Start testing...")
            le_ro_n_episodes = ro_n_episodes
            for log_beta in (args.log_betas_eval if args.algo == "soft_q" else [0]):
                all_rewards = []
                all_ep_lengths = []
                cum_obs = None
                traj_dict = defaultdict(lambda:0)
                for s in range(args.n_episodes_eval):
                    current_rewards = []
                    eval_vec_env = DummyVecEnv(env_fn=make_env_fn, batch_size=1) # NEW!!!
                    eval_obs = eval_vec_env.reset()["player1"].unsqueeze(0)
                    if cum_obs is None:
                        cum_obs = eval_obs.clone()
                    else:
                        cum_obs += eval_obs # count states
                    other_dct = {}
                    if args.algo == "soft_q":
                        other_dct["log_beta"] = th.FloatTensor([[log_beta]]).to(args.net_device)
                    eval_t = 0
                    act_lst = []
                    max_eval_steps = 3000
                    while True:
                        if args.input_t:
                            other_dct["t"] = th.FloatTensor([[eval_t]])
                        q_values = models["net"](eval_obs.to(args.net_device), other_dct)["out"]
                        avail_actions = eval_vec_env.get_available_actions()["player1"].to(args.net_device)

                        if args.algo == "soft_q":
                            policy = masked_softmax(q_values,
                                                    mask=(avail_actions == 0),
                                                    beta=th.exp(other_dct["log_beta"]),
                                                    dim=-1)
                            next_action = th.multinomial(policy.squeeze(0), 1)
                        else:
                            q_values[~avail_actions] = -1E20
                            next_action = th.argmax(q_values, -1, keepdim=True).detach()

                        act_lst.append(next_action[0].item())
                        eval_obs, reward, done, info = eval_vec_env.step(next_action)
                        eval_obs = eval_obs["player1"]
                        cum_obs += eval_obs
                        current_rewards.append(reward.item())
                        eval_t += 1
                        if False:
                            eval_vec_env.render()
                        if done[0] or eval_t > max_eval_steps:
                            all_rewards.append(np.sum(current_rewards))
                            all_ep_lengths.append(eval_t)
                            traj_dict[tuple(act_lst)] += 1
                            break

                if args.algo == "soft_q":
                    print("Mean episode reward test (log beta: {:.2f}): {:.2f} +- {:.2f} len: {:.2f} +- {:.2f} n_trajs: {:.2f}".format(log_beta,
                                                                                                                       np.mean(all_rewards),
                                                                                                                       np.std(all_rewards),
                                                                                                                       np.mean(all_ep_lengths),
                                                                                                                       np.std(all_ep_lengths),
                                                                                                                       len(traj_dict.values())))
                    print("CUM_OBS: ")
                    print(cum_obs)
                else:
                    print("Mean episode reward test: {:.2f} len: {:.2f}".format(np.mean(all_rewards), np.mean(all_ep_lengths)))


            score = np.mean(all_rewards)
            if score >= best_score:
                best_score = score
                print("Saving model...")
                with open(os.path.join(path, "model_ep{}.pkl".format(ro_n_episodes)), "wb") as f:
                    pickle.dump(args, f)
                with open(os.path.join(path, "model_ep{}.json".format(ro_n_episodes)), "w") as f:
                    json.dump(vars(args), f, ensure_ascii=True, indent=4, sort_keys=True)
                th.save(models["net"], os.path.join(path, "model_ep{}_score{}.pt".format(ro_n_episodes, score)))
        ################################ EVAL STOP #####################################################################
