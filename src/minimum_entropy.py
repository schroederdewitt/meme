import argparse
from collections import defaultdict
from functools import partial
#from minimum_entropy_utils import minimum_entropy_coupling
from minimum_entropy_sparse_utils import minimum_entropy_coupling
import numpy as np
import pickle
from pprint import pprint
import torch as th
import time

from utils import masked_softmax
from envs import Simple4WayGrid, Simple4WayGridPrivateObs, DummyVecEnv

# Load saved model, and parameter file
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, help="learning algorithm")
parser.add_argument("--episodes-per-token", type=int, default=10, help="learning algorithm")
args = parser.parse_args()
pprint(vars(args))

if args.model_path == "pong":
    obs_space = th.load("results_prio/pong_obsspace.pt")
    model_args_dct = {"env": "pong",
                      "bs_rollout": 1,
                      "algo": "soft_q",
                      "net_device" : "cuda:0"}

    net = PongNet(obs_space).to(model_args_dct["net_device"])
    state_dict = th.load("results_prio/pong_weights.pt")
    suff1 = "q_net.features_extractor"
    suff2 = "q_net.q_net"
    state_dict = {("net" + k[len(suff1):] if k[:len(suff1)] == suff1 else k):v for k, v in state_dict.items()}
    state_dict = {("net" + k[len(suff2):] if k[:len(suff2)] == suff2 else k):v for k, v in state_dict.items()}
    suff3 = "q_net_target.features_extractor"
    for k in list(state_dict.keys()):
        if k[:len(suff3)] == suff3:
            del state_dict[k]
    suff4 = "q_net_target.q_net"
    for k in list(state_dict.keys()):
        if k[:len(suff4)] == suff4:
            del state_dict[k]
    state_dict["linear_q.weight"] = state_dict.pop("net.0.weight")
    state_dict["linear_q.bias"] = state_dict.pop("net.0.bias")
    net.load_state_dict(state_dict)
    # th.load(net.state_dict(), PATH)
    from types import SimpleNamespace
    model_args = SimpleNamespace(**model_args_dct)

elif args.model_path in ["pongfull", "pongall"]:
    obs_space = th.load("results_prio/pong_obsspace.pt")
    model_args_dct = {"env": "pong" if args.model_path != "pongall" else "pongall",
                      "bs_rollout": 1,
                      "algo": "soft_q",
                      "net_device" : "cuda:0"}

    net = PongNet(obs_space).to(model_args_dct["net_device"])
    state_dict = th.load("results_prio/pongfull_weights.pt")
    suff1 = "q_net.features_extractor"
    suff2 = "q_net.q_net"
    state_dict = {("net" + k[len(suff1):] if k[:len(suff1)] == suff1 else k):v for k, v in state_dict.items()}
    state_dict = {("net" + k[len(suff2):] if k[:len(suff2)] == suff2 else k):v for k, v in state_dict.items()}
    suff3 = "q_net_target.features_extractor"
    for k in list(state_dict.keys()):
        if k[:len(suff3)] == suff3:
            del state_dict[k]
    suff4 = "q_net_target.q_net"
    for k in list(state_dict.keys()):
        if k[:len(suff4)] == suff4:
            del state_dict[k]
    state_dict["linear_q.weight"] = state_dict.pop("net.0.weight")
    state_dict["linear_q.bias"] = state_dict.pop("net.0.bias")
    net.load_state_dict(state_dict)

    from types import SimpleNamespace
    model_args = SimpleNamespace(**model_args_dct)

elif args.model_path == "breakout":
    obs_space = th.load("results_prio/pong_obsspace.pt")
    model_args_dct = {"env": "breakout",
                      "bs_rollout": 1,
                      "algo": "soft_q",
                      "net_device" : "cuda:0"}

    net = PongNet(obs_space, n_actions=4).to(model_args_dct["net_device"])
    state_dict = th.load("results_prio/breakout/breakout_weights.pt")
    suff1 = "q_net.features_extractor"
    suff2 = "q_net.q_net"
    state_dict = {("net" + k[len(suff1):] if k[:len(suff1)] == suff1 else k):v for k, v in state_dict.items()}
    state_dict = {("net" + k[len(suff2):] if k[:len(suff2)] == suff2 else k):v for k, v in state_dict.items()}
    suff3 = "q_net_target.features_extractor"
    for k in list(state_dict.keys()):
        if k[:len(suff3)] == suff3:
            del state_dict[k]
    suff4 = "q_net_target.q_net"
    for k in list(state_dict.keys()):
        if k[:len(suff4)] == suff4:
            del state_dict[k]
    state_dict["linear_q.weight"] = state_dict.pop("net.0.weight")
    state_dict["linear_q.bias"] = state_dict.pop("net.0.bias")
    net.load_state_dict(state_dict)

    from types import SimpleNamespace
    model_args = SimpleNamespace(**model_args_dct)

elif args.model_path == "breakout_sb3":
    obs_space = th.load("results_prio/pong_obsspace.pt")
    model_args_dct = {"env": "breakout",
                      "bs_rollout": 1,
                      "algo": "soft_q",
                      "net_device" : "cuda:0"}

    net = PongNet(obs_space, n_actions=4).to(model_args_dct["net_device"])
    state_dict = th.load("results_prio/breakout/breakout_weights.pt")
    suff1 = "q_net.features_extractor"
    suff2 = "q_net.q_net"
    state_dict = {("net" + k[len(suff1):] if k[:len(suff1)] == suff1 else k):v for k, v in state_dict.items()}
    state_dict = {("net" + k[len(suff2):] if k[:len(suff2)] == suff2 else k):v for k, v in state_dict.items()}
    suff3 = "q_net_target.features_extractor"
    for k in list(state_dict.keys()):
        if k[:len(suff3)] == suff3:
            del state_dict[k]
    suff4 = "q_net_target.q_net"
    for k in list(state_dict.keys()):
        if k[:len(suff4)] == suff4:
            del state_dict[k]
    state_dict["linear_q.weight"] = state_dict.pop("net.0.weight")
    state_dict["linear_q.bias"] = state_dict.pop("net.0.bias")
    net.load_state_dict(state_dict)

    from types import SimpleNamespace
    model_args = SimpleNamespace(**model_args_dct)

elif args.model_path == "tennis":
    obs_space = th.load("results_prio/pong_obsspace.pt")
    model_args_dct = {"env": "tennis",
                      "bs_rollout": 1,
                      "algo": "soft_q",
                      "net_device" : "cuda:0"}

    net = PongNet(obs_space, n_actions=18).to(model_args_dct["net_device"])
    state_dict = th.load("results_prio/tennis/tennis_weights.pt")
    suff1 = "q_net.features_extractor"
    suff2 = "q_net.q_net"
    state_dict = {("net" + k[len(suff1):] if k[:len(suff1)] == suff1 else k):v for k, v in state_dict.items()}
    state_dict = {("net" + k[len(suff2):] if k[:len(suff2)] == suff2 else k):v for k, v in state_dict.items()}
    suff3 = "q_net_target.features_extractor"
    for k in list(state_dict.keys()):
        if k[:len(suff3)] == suff3:
            del state_dict[k]
    suff4 = "q_net_target.q_net"
    for k in list(state_dict.keys()):
        if k[:len(suff4)] == suff4:
            del state_dict[k]
    state_dict["linear_q.weight"] = state_dict.pop("net.0.weight")
    state_dict["linear_q.bias"] = state_dict.pop("net.0.bias")
    net.load_state_dict(state_dict)

    from types import SimpleNamespace
    model_args = SimpleNamespace(**model_args_dct)

else:
    net = th.load(args.model_path+".pt")
    with open(args.model_path+".pkl", "rb") as f:
        model_args = pickle.load(f)

########################################################################################################################
# ARGPARSE -- STOP
########################################################################################################################

# Create environment
env_registry = {"simple_4way_grid": Simple4WayGrid,
                "mini_grid": None,
                "simple_4way_private_obs": Simple4WayGridPrivateObs,
                }


assert model_args.env in env_registry, "Environment {} not in registry!".format(model_args.env)

make_env_fn = None
if model_args.env in ["simple_grid", "simple_4way_grid"]:
    def _make_env_fn():
        env = env_registry[model_args.env](max_steps=model_args.env_arg_max_steps,
                                     grid_dim=model_args.env_arg_grid_dim)
        env.max_steps = model_args.env_arg_max_steps
        return env

elif model_args.env in ["simple_4way_private_obs"]:
    def _make_env_fn():
        token_dist = None
        if args.token_dist in ["uniform"]:
            rand_cats = np.ones(model_args.token_n)
            rand_cats /= rand_cats.sum()
            token_dist = partial(np.random.choice, a=np.arange(model_args.token_n), size=1, p=rand_cats)
        elif args.token_dist in ["random"]:
            rand_cats = np.random.rand(model_args.token_n)
            rand_cats /= rand_cats.sum()
            token_dist = partial(np.random.choice, a=np.arange(model_args.token_n), size=1, p=rand_cats)
        else:
            pass
        env = env_registry[args.env](max_steps=model_args.env_arg_max_steps,
                                     grid_dim=model_args.env_arg_grid_dim,
                                     private_obs_dist=token_dist,
                                     token_n=model_args.token_n,
                                     private_obs_dim=model_args.token_dim,
                                     token_reward_factor=model_args.env_args_token_reward_factor)
        env.max_steps = args.env_arg_max_steps
        return env

elif model_args.env in ["pong", "breakout", "tennis", "pongall"]:
    def _make_env_fn():
        env = env_registry[model_args.env]()
        return env


make_env_fn = _make_env_fn

# Create vec envs
vec_env = DummyVecEnv(env_fn=make_env_fn, batch_size=model_args.bs_rollout)
eval_vec_env = DummyVecEnv(env_fn=make_env_fn, batch_size=1)


def please_plot_token(plot_dims, pred_token, pz_dct, chunks, prefix):
    res_arr_bin = np.zeros(plot_dims).flatten()
    bitctr = 0
    for i, (p, c) in enumerate(zip(pred_token, chunks)):
        bitlen = int(np.log2(c))
        rst = p
        token_arr = [0]*bitlen
        for j in reversed(range(bitlen)):
            if rst >= 2**j:
                token_arr[j] = 1
                rst -= 2**j
        res_arr_bin[bitctr:bitctr+bitlen] = np.array(token_arr)
        bitctr += bitlen

    color_map = plt.cm.get_cmap('Greys').reversed()
    res_arr_bin = res_arr_bin.reshape(plot_dims)
    plt.imshow(res_arr_bin, cmap=color_map,  interpolation='nearest')
    _hash = uuid.uuid4().hex[:6]
    plt.axis('off')
    plt.savefig("{}__posteriormax__{}.png".format(prefix, _hash))
    plt.show()

    res_arr_bin_post = np.zeros(plot_dims).flatten()
    bitctr = 0
    for i, c in enumerate(chunks):
        post = pz_dct[i][-1]
        bitlen = int(np.log2(c))
        for j, p in enumerate(post):
            rst = j
            token_arr = [0]*bitlen
            for j in reversed(range(bitlen)):
                if rst >= 2**j:
                    token_arr[j] = 1
                    rst -= 2**j
            res_arr_bin_post[bitctr:bitctr+bitlen] += np.array(token_arr)*p.item()
        bitctr += bitlen

    res_arr_bin_post = res_arr_bin_post.reshape(plot_dims)
    plt.imshow(res_arr_bin_post, cmap=color_map,  interpolation='nearest')
    plt.axis('off')
    plt.savefig("{}__posterior__{}.png".format(prefix, _hash))
    plt.show()
    pass

from matplotlib import pyplot as plt
def load_image_to_token(path, chunks):
    image = np.rint(plt.imread(path)).astype(np.int32)[:,:,0]
    image_flat = image.flatten()
    token_lst = []
    ctr = 0
    for i, c in enumerate(chunks):
        bitlen = int(np.log2(c))
        img_seq = image_flat[ctr:ctr+bitlen]
        token_lst.append(np.sum(2**np.array(list(range(bitlen)))* img_seq))
        ctr += bitlen
    assert ctr == image_flat.shape[0], "Image does not have the right size!"
    return token_lst


#### OFFICIAL PARAMETERS ######
# Gridworld 4x4:
#n_tokens_list = [0, 2, 5, 10, 20, 50]
# noise_p = [0.0, 0.05, 0.25]
# log_betas_eval = [1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0]

n_tokens_list = [2,4,8,16]
log_betas_eval = [0.1, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 5.0, 7.0]
noise_p = [0.0, 0.05, 0.25]
n_draws_per_token = 20 #20
total_token_draws = 20 #int(10E4)


n_sims = len(n_tokens_list) # how many different token distributions should be considered

chunking_dict = {(2,8*32): [16]*64, #[16]*64, # 32 bytes!
                 (2,14*14): [256]*37,#[16]*74,
                 (2,12*12): [16]*36,
                 (2,128): [16]*32,
                 (2,100): [10]*10,
                 (2,64): [16]*16,
                 2**32: [16]*8,
                 2**24: [16]*6,
                 2**16: [16]*4,
                 32**3: [32]*3,
                 10000: [10,10,10,10],
                 1024: [32, 32],
                 1000: [10,10,10],
                 100: [10,10],
                 50: [5,10],
                 49: [7,7],
                 10: [2,5],
                 2: [3,4]} # maps numbers of tokens to chunkings

import uuid
import dill
uid = uuid.uuid4().hex[:6]

recalc_joint_p = True # Only set False if you have checked joint p is recalculated entirely faithfully!
recalc_q_values = True # Only set False if you have checked q values are recalculated entirely faithfully!
use_minimum_entropy = True # Only set this to False if you want to have *pristine* performance
use_noise_correction = True
correct_posterior_collapse = True
noise_actions_player1 = True
noise_actions_player2 = False
decode_in_encoder = True # True # saves massive compute!
use_message_chunking = False # chunk messages!
message_chunking_traversal_mode = "cycle" # cycle through messages
message_chunking_entropy_threshold = 0.009 # 1E-4 #1E-2
render_env = False #False #
save_env_render = False
mec_mode = "dense" #"sparse" #"dense"
mec_atol = 1E-7
warning_atol = 1E-5
n_tokens_memory_threshold = 2**24
use_fixed_token = False #True #True
fixed_token_fn = partial(load_image_to_token, path="./yingyang.png")
plot_dims = (16,16)
plot_token = False #True# True # token is plotted!
token_plots_path = "token_plots/token_plots_2**256"
import os
if not os.path.exists(token_plots_path):
    os.makedirs(token_plots_path)
token_plot_fn = partial(please_plot_token, plot_dims=plot_dims)
plot_token_dims = plot_dims

if not use_minimum_entropy:
    print("WARNING: use_minimum_entropy has been set to False!")

# Note: We have adjusted token_rollouts to reflect uniform probabilities!

old_time = time.time()
for ns in range(n_sims):
    for noise in noise_p:
        results_dct = {}
        n_tokens = n_tokens_list[ns]
        if n_tokens == 0:
            n_token_draws = total_token_draws
            token_rollouts = np.array([0.0]*n_draws_per_token)
        elif not isinstance(n_tokens, tuple) and n_tokens < n_tokens_memory_threshold:
            token_rollouts = np.repeat(np.array(list(range(n_tokens))),
                                       n_draws_per_token)  # UNIFORM MESSAGE PROBABILITY!
            np.random.shuffle(token_rollouts)
            n_token_draws = n_draws_per_token * n_tokens
        else:
            print("Exceeding token draw memory limit... will just sample tokens randomly with replacement!")
            n_token_draws = total_token_draws


        results_dct[n_tokens] = {}
        log_betas_tokens = defaultdict(lambda: [])
        for log_beta in (log_betas_eval if model_args.algo == "soft_q" else [0]):
            print("Number of tokens: {} Noise: {} LogBeta: {}".format(n_tokens, noise, log_beta))
            first_right_ctr_dct = defaultdict(lambda: [])
            first_right_ctr_dct_noise = defaultdict(lambda: [])
            best_entropy_dct = defaultdict(lambda: [])
            additive_gap_dct = defaultdict(lambda: [])
            token_success_rate_dct = defaultdict(lambda: [])
            token_success_rate_noise_dct = defaultdict(lambda: [])
            traj_dict_dct = defaultdict(lambda: [])
            cum_obs_all = []
            all_rewards = []
            all_ep_lengths = []
            traj_dict = defaultdict(lambda: 0)
            chunks = None
            if use_message_chunking:
                chunks = chunking_dict.get(n_tokens, [n_tokens])
                print("Using message chunking for ntokens: {} with chunks: {}".format(n_tokens, chunks))

            for nt in range(n_token_draws):
                if not use_fixed_token:
                    if not isinstance(n_tokens, tuple) and n_tokens < n_tokens_memory_threshold:
                        token = token_rollouts[nt] # better than sampling!
                    else:
                        token = [np.random.randint(c) for c in chunks]
                else:
                    print("Loading fixed token...")
                    token = fixed_token_fn(chunks = chunks)

                token_chunks = None
                if use_message_chunking:
                    active_chunk_ids_dct = {k: 1 for k in range(len(chunks))} # chunk ids that are active

                    if not isinstance(token, list):
                        token_chunks = [0 for _ in chunks]
                        ctmp = 1
                        for i, c in enumerate(chunks):
                            ctmp *= c
                            if ctmp > token: # tokens start at 0!
                                tj = token
                                for j in reversed(range(i+1)):
                                    token_chunks[j] = int(tj // np.prod(chunks[:j]))
                                    tj = tj %  np.prod(chunks[:j])
                                break
                        print("Token {} was chunked into: {}".format(token, token_chunks))
                    else:
                        token_chunks = token
                cum_obs = None
                for s in range(1): #args.episodes_per_token):
                    current_rewards = []
                    eval_obs = eval_vec_env.reset()["player1"]
                    if cum_obs is None:
                        cum_obs = eval_obs.clone()
                    else:
                        cum_obs += eval_obs  # count states
                    other_dct = {}
                    if model_args.algo == "soft_q":
                        other_dct["log_beta"] = th.FloatTensor([[log_beta]]).to(model_args.net_device)
                    eval_t = 0
                    act_lst = []
                    state_lst = [eval_obs.clone()]
                    avail_actions_lst = []
                    joint_p_lst = []
                    q_values_lst = []
                    if isinstance(n_tokens, tuple) or n_tokens > 0:
                        if use_message_chunking:
                            p_z_noise_dct = {k:th.DoubleTensor([1.0] * chunks[k]) for k in range(len(chunks))}  # uniform prior initially
                            p_z_noise_dct = {k:v/v.sum() for k, v in p_z_noise_dct.items()}

                            p_z_noise_lst_dct = {k:[] for k, v in p_z_noise_dct.items()}
                        else:
                            p_z = th.DoubleTensor([1.0] * n_tokens)  # uniform prior initially
                            p_z /= p_z.sum()
                            p_z_noise = th.DoubleTensor([1.0] * n_tokens)  # uniform prior initially
                            p_z_noise /= p_z_noise.sum()
                            p_z_noise_lst = []
                        cond_lst = []
                    while True:
                        if use_message_chunking:
                            if message_chunking_traversal_mode == "cycle":
                                active_chunk_id_keys = [k for k in active_chunk_ids_dct.keys() if active_chunk_ids_dct[k]==1.0]
                                # the active chunk is the one with the highest posterior entropy
                                # calculate posterior entropy over all posteriors

                            else:
                                assert False, "Unknown message chunking traversal mode!"
                        other_dct["t"] = th.FloatTensor([[eval_t]])
                        q_values = net(eval_obs.to(model_args.net_device), other_dct)["out"]
                        q_values_lst.append(q_values)
                        avail_actions = eval_vec_env.get_available_actions()["player1"].to(model_args.net_device)
                        avail_actions_lst.append(avail_actions.clone())

                        if model_args.algo == "soft_q":
                            policy = masked_softmax(q_values,
                                                    mask=(avail_actions == 0),
                                                    beta=th.exp(other_dct["log_beta"]),
                                                    dim=-1)
                            policy_probs = policy.to("cpu")
                            if not use_minimum_entropy or n_tokens == 0.0 or (use_message_chunking and not len(active_chunk_id_keys)):
                                if noise_actions_player1:
                                    s = np.random.random()
                                    if s <= noise:
                                        next_action = th.multinomial(avail_actions.float(), 1).squeeze(0)
                                    else:
                                        next_action = th.multinomial(policy.squeeze(0), 1)
                            else:
                                # Step 1: find p hat q
                                with th.no_grad():
                                    if use_message_chunking:
                                        p_z_noise = p_z_noise_dct[active_chunk_id]
                                    joint_p = minimum_entropy_coupling(p_z_noise.type(th.float64),
                                                                       policy_probs.squeeze(0).type(th.float64),
                                                                       mode=mec_mode, algo_atol=mec_atol,
                                                                       warning_atol=warning_atol)
                                    if joint_p["warnings"]:
                                        print("WARNINGS (marginalisation) found in MEC calculation... p_error: {:.1e} q_error: {:.1e}".format(joint_p["p_error"],
                                                                                                                                      joint_p["q_error"]))

                                if not use_noise_correction:
                                    # Detect posterior collapse
                                    M_sum = joint_p["M"].sum(1)
                                    if (M_sum == 0.0).sum() != 0.0: # i.e. there are rows of all zeros
                                        print("WARNING: Zero rows found in M!")
                                        if correct_posterior_collapse:
                                            joint_p["M"][(M_sum == 0.0).unsqueeze(1).repeat(1, joint_p["M"].shape[1])] = \
                                                1.0/float(joint_p["M"].shape[1])
                                            print("Stabilized posterior!")
                                        else:
                                            print("Did not stabilize posterior.")

                                joint_p_lst.append(joint_p)

                                if noise_actions_player1:
                                    s = np.random.random()
                                    if s <= noise:
                                        next_action = th.multinomial(avail_actions.float(), 1)# .squeeze(0)
                                    else:
                                        if use_message_chunking:
                                            if (joint_p["M"][token_chunks[active_chunk_id]].sum() <= 0.0):
                                                print("Sum ZERO ERROR: Recovering by following the policy")
                                                next_action = th.multinomial(policy.squeeze(0), 1)
                                                p_z_noise = th.DoubleTensor([1.0] * n_tokens)
                                            else:
                                                next_action = th.multinomial(joint_p["M"][token_chunks[active_chunk_id]].squeeze(0), 1)
                                        else:
                                            if (joint_p["M"][token].sum() <= 0.0):
                                                print("Sum ZERO ERROR: Recovering by following the policy")
                                                next_action = th.multinomial(policy.squeeze(0), 1)
                                                p_z_noise = th.DoubleTensor([1.0] * n_tokens)
                                            else:
                                                next_action = th.multinomial(joint_p["M"][token].squeeze(0), 1)
                                if use_noise_correction:
                                    num_actions = joint_p["M"].shape[1]
                                    p_z_mid_a = joint_p["M"] / joint_p["M"].sum(dim=0)
                                    p_z_mid_a[th.isnan(p_z_mid_a)] = 0
                                    p_o_mid_a_clean = th.zeros(num_actions, num_actions)
                                    avail = avail_actions.squeeze().to("cpu")
                                    p_o_mid_a_clean[avail, avail] = 1
                                    p_o_mid_a_corrupted = th.outer(avail, avail).int()
                                    p_o_mid_a_corrupted = p_o_mid_a_corrupted / p_o_mid_a_corrupted.sum(dim=0)
                                    p_o_mid_a = (1 - noise) * p_o_mid_a_clean + noise * p_o_mid_a_corrupted
                                    p_a = joint_p["M"].sum(dim=0)
                                    p_a_mid_o = (p_o_mid_a * p_a[:, None]) / (p_o_mid_a * p_a[:, None]).sum(dim=0)
                                    p_a_mid_o[th.isnan(p_a_mid_o)] = 0
                                    p_z_mid_o = p_z_mid_a @ p_a_mid_o
                                    p_z_noise = p_z_mid_o[:, next_action.item()]
                                else:
                                    p_z_noise = joint_p["M"][:, next_action[0].item()] / joint_p["M"][:, next_action[0].item()].sum()

                                if p_z_noise.sum() == 0.0:
                                    print("WARNING: p_z_noise.sum() == 0.0 - resetting!")
                                    p_z_noise = th.DoubleTensor([1.0] * n_tokens)
                                p_z_noise /= p_z_noise.sum()

                                if use_message_chunking:
                                    p_z_noise_lst_dct[active_chunk_id].append(p_z_noise)
                                    p_z_noise_dct[active_chunk_id] = p_z_noise

                                    # check if posterior entropy is below threshold, if so deactivate active_chunk_id
                                    ent = -th.sum(th.log2(p_z_noise[p_z_noise!=0.0])*p_z_noise[p_z_noise!=0.0])
                                    if message_chunking_entropy_threshold >= 0.0 and len(active_chunk_id_keys) and ent < message_chunking_entropy_threshold:
                                        active_chunk_ids_dct[active_chunk_id] = 0.0
                                        active_chunk_id_keys = [k for k in active_chunk_ids_dct.keys() if
                                                                active_chunk_ids_dct[k] == 1.0]
                                        print("Active chunk id {} was deactivated as entropy fell to {}/{}... only {} left active!".format(active_chunk_id,
                                                                                                                      ent,
                                                                                                                      message_chunking_entropy_threshold,
                                                                                                                      len(active_chunk_id_keys)))

                                else:
                                    p_z_noise_lst.append(p_z_noise)

                                # additive_gap_dct[token_chunks[active_chunk_id].item()].append(joint_p["additive_gap"])
                                try:
                                    cond_lst.append(joint_p["M"][:, next_action[0].item()] / joint_p["M"][:, next_action[0].item()].sum())
                                except Exception as e:
                                    l = 1
                                    pass
                            # Finished minimum entropy coupling
                        else:
                            q_values[~avail_actions] = -1E20
                            next_action = th.argmax(q_values, -1, keepdim=True).detach()

                        act_lst.append(next_action[0].item())
                        eval_obs, reward, done, info = eval_vec_env.step(next_action)
                        eval_obs = eval_obs["player1"].squeeze(0)
                        state_lst.append(eval_obs)
                        cum_obs += eval_obs
                        current_rewards.append(reward)
                        eval_t += 1

                        if render_env:
                            img = eval_vec_env.render()
                        if save_env_render:
                            hsh = uuid.uuid4().hex[:2]
                            ent = -(th.log2(policy_probs[policy_probs!=0.0])*policy_probs[policy_probs!=0.0]).sum()
                            eval_vec_env.envs[0].env.venv.envs[0].ale.saveScreenPNG(str.encode('screens2/PONG_eval_t{}_logbeta{}_ent{:.2e}_hsh{}.png'.format(
                                eval_t,
                                log_beta,
                                ent,
                                hsh
                            )))
                        if done[0]:
                            new_time = time.time()
                            print("done - {}/{} (succ rate: {} rewards: {} time: {})".format(nt,
                                                                        n_token_draws,
                                                                        np.mean(np.array(token_success_rate_noise_dct[log_beta])),
                                                                        np.mean(all_rewards),
                                                                        new_time - old_time),
                                                                        )
                            old_time = new_time
                            all_rewards.append(np.sum(current_rewards))
                            all_ep_lengths.append(eval_t)
                            traj_dict[tuple(act_lst)] += 1
                            if (n_tokens == 0.0):
                                break

                            if decode_in_encoder:
                                if use_message_chunking:
                                    if not isinstance(token, list):
                                        pred_token_noise = 0
                                        for ci, c in enumerate(chunks):
                                            p_z_noise = p_z_noise_lst_dct[ci][-1]
                                            ptn = th.argmax(p_z_noise).item()
                                            pred_token_noise += (np.prod(chunks[:ci]) if ci else 1.0) * ptn
                                    else:
                                        pred_token_noise = [th.argmax(p_z_noise_lst_dct[ci][-1]).item() for ci, c in enumerate(chunks)]
                                else:
                                    p_z_noise = p_z_noise_lst[-1]
                                    pred_token_noise = th.argmax(p_z_noise).item()
                                if not isinstance(pred_token_noise, list):
                                    token_success_rate_noise_dct[log_beta].append(int(pred_token_noise == token.item()))
                                else:
                                    token_success_rate_noise_dct[log_beta].append(int(np.sum(np.array(token) - np.array(pred_token_noise)) == 0))
                                if plot_token:
                                    print("Plotting the token now...")
                                    token_plot_fn(pred_token=pred_token_noise,
                                                  pz_dct=p_z_noise_lst_dct,
                                                  chunks=chunks,
                                                  prefix="{}/{}_p{}_logbeta{}_rew{}".format(token_plots_path, args.model_path, noise, log_beta, np.sum(current_rewards)))
                                break
                            else:
                                assert False, "!decode_in_encoder Option currently not available!"
                                # Player 2: Decode action-observation history
                                p_z_noise_lst2 = []
                                p_z_noise = th.DoubleTensor([1.0] * n_tokens)  # uniform prior initially
                                p_z_noise /= p_z_noise.sum()
                                eval_t = 0
                                first_right_ctr = 0.0
                                first_right_ctr_noise = 0.0
                                for i in range(len(state_lst)-1):
                                    other_dct["t"] = th.FloatTensor([[eval_t]])
                                    state = state_lst[i]
                                    avail_actions = avail_actions_lst[i]
                                    action = act_lst[i]
                                    if noise_actions_player2:
                                        s = np.random.random()
                                        if s <= noise:
                                            action = th.multinomial(avail_actions.float(), 1).item()
                                    if recalc_q_values or  n_tokens == 0.0 or not use_minimum_entropy:
                                        q_values = net(state.to(model_args.net_device), other_dct)["out"]
                                    else:
                                        q_values = q_values_lst[i]
                                    policy = masked_softmax(q_values,
                                                            mask=(avail_actions == 0),
                                                            beta=th.exp(other_dct["log_beta"]),
                                                            dim=-1)
                                    policy_probs = policy.to("cpu")
                                    if recalc_joint_p or (not use_minimum_entropy):
                                        with th.no_grad():
                                            if th.sum(p_z_noise[p_z_noise != p_z_noise]) != 0.0:
                                                a = 5
                                            joint_p = minimum_entropy_coupling(p_z_noise.type(th.float64),
                                                                               policy_probs.squeeze(0).type(th.float64))

                                            if joint_p["M"][token].sum() == 0.0:
                                                # THIS is a weird instability fix that occurs some of the time if one of the
                                                # posterior modes collapses
                                                print("WARNING: POSTERIOR COLLAPSE!")
                                                p_z_noise = th.DoubleTensor([1.0] * n_tokens)
                                                p_z_noise /= p_z_noise.sum()
                                                joint_p["M"][token] += 1 / float(len(joint_p["M"][token]))
                                    else:
                                        joint_p = joint_p_lst[i]
                                    p_z = p_z * joint_p["M"][:, action] # / joint_p[:, action].sum())
                                    p_z /= p_z.sum()
                                    if joint_p["M"][:, action].sum() == 0.0:
                                        print("WARNING: M Sum Error. Likely source: An available action (sampled through noise) has a very small q value. Reconstructing M...")
                                        joint_p["M"][:, action] += 1.0 # will be renormalizing itself anyway

                                    if use_noise_correction:
                                        p_z_noise = (1 - noise) * joint_p["M"][:, action]/joint_p["M"][:, action].sum() + \
                                                    noise*p_z_noise / p_z_noise.sum()
                                    else:
                                        p_z_noise = joint_p["M"][:, action]/joint_p["M"][:, action].sum()
                                    if p_z_noise.sum() == 0.0:
                                        print("WARNING: p_z_noise.sum() == 0.0 - resetting!")
                                        p_z_noise = th.DoubleTensor([1.0] * n_tokens)
                                    p_z_noise /= p_z_noise.sum()
                                    if th.sum(p_z_noise[p_z_noise != p_z_noise]) != 0.0:
                                        a = 5
                                    p_z_noise_lst2.append(p_z_noise)
                                    eval_t += 1
                                    pred_token = th.argmax(p_z).item()
                                    pred_token_noise = th.argmax(p_z_noise).item()
                                    best_entropy_dct[token.item()].append(joint_p["best_entropy"])
                                    additive_gap_dct[token.item()].append(joint_p["additive_gap"])
                                    if not first_right_ctr and pred_token == token.item():
                                        first_right_ctr = eval_t
                                    if not first_right_ctr_noise and pred_token_noise == token.item():
                                        first_right_ctr_noise = eval_t

                                pred_token = th.argmax(p_z).item()
                                if pred_token == token.item():
                                    first_right_ctr_dct[token.item()].append(first_right_ctr)
                                token_success_rate_dct[log_beta].append(int(pred_token == token.item()))
                                token_success_rate_noise_dct[log_beta].append(int(pred_token_noise == token.item()))

                                break
                traj_dict_dct[token.item() if not isinstance(token, list) else tuple(token)].append(len(list(traj_dict.values())))
                cum_obs_all.append(cum_obs)

            token_success_rate = np.mean(np.array(token_success_rate_dct[log_beta]))
            token_success_rate_noise = np.mean(np.array(token_success_rate_noise_dct[log_beta]))
            results_dct[n_tokens][log_beta] = {"token_success_rate": token_success_rate,
                                               "token_success_rate_noise": token_success_rate_noise,
                                               "token_success_rate_dct": token_success_rate_dct,
                                               "token_success_rate_noise_dct": token_success_rate_noise_dct,
                                               "all_rewards": all_rewards,
                                               "all_ep_lengths": all_ep_lengths,
                                               "game_reward_mean": np.mean(all_rewards),
                                               "game_reward_std": np.std(all_rewards),
                                               "episode_length_mean": np.mean(all_ep_lengths),
                                               "episode_length_std": np.std(all_ep_lengths),
                                               "n_trajectories": traj_dict_dct,
                                               "first_right_ctr_dct": first_right_ctr_dct,
                                               "first_right_ctr_dct_noise": first_right_ctr_dct_noise,
                                               "best_entropy": best_entropy_dct,
                                               "cum_obs": th.stack(cum_obs_all, axis=0).sum(axis=0).squeeze(),
                                               "noise": noise}
            print("Log beta {} success rate: {} success rate noise: {} ({} tokens)".format(log_beta,
                                                                                           token_success_rate,
                                                                                           token_success_rate_noise,
                                                                                           n_tokens))

            if model_args.algo == "soft_q":
                print(
                    "Mean episode reward test (log beta: {:.2f}): {:.2f} +- {:.2f} len: {:.2f} +- {:.2f} n_trajs: {:.2f}".format(
                        log_beta,
                        np.mean(all_rewards),
                        np.std(all_rewards),
                        np.mean(all_ep_lengths),
                        np.std(all_ep_lengths),
                        np.mean([v[-1] for v in traj_dict_dct.values()])))
                print("CUM_OBS: ")
                print(th.stack(cum_obs_all, axis=0).sum(axis=0).squeeze())
            else:
                print("Mean episode reward test: {:.2f} len: {:.2f}".format(np.mean(all_rewards), np.mean(all_ep_lengths)))


        fname = "results_ntok{}_noise{}_{}.pkl".format(n_tokens, noise, uid)
        with open(fname, "wb") as f:
            dill.dump(results_dct, f)

print("SAVED as: {}".format(fname))
