from functools import reduce
import operator
from torch import nn
import torch as th
import torch.nn.functional as F

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

class MLPLSTMNetDeep(nn.Module):
    def __init__(self, n_actions, input_shape, other_shape_dct=None, rnn_hidden_dim=64):
        super(MLPLSTMNetDeep, self).__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.other_shape_dct = other_shape_dct
        self.rnn_hidden_dim = rnn_hidden_dim
        self.other_shape_lin = sum([reduce(operator.mul, v) for v in other_shape_dct.values()]) if other_shape_dct != None else 0
        self.fc0 = nn.Sequential(nn.Linear(reduce(operator.mul, self.input_shape) + self.other_shape_lin, self.rnn_hidden_dim),
                                 nn.ReLU())
        self.rnn_cell = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.fc1 = nn.Sequential(nn.Linear(self.rnn_hidden_dim, 64),
                                 nn.ReLU())
        self.fc2 = nn.Linear(64, self.n_actions)

    def forward(self, x, other_dct=None, is_sequence=False, hidden=None):
        if is_sequence:
            z = x.view(x.shape[0], x.shape[1], -1)
            if other_dct is not None:
                z = th.cat(
                    [z] + [other_dct[k].to(x.device).view(other_dct[k].shape[0], other_dct[k].shape[1], -1) for k in sorted(list(other_dct.keys()))], -1)
        else:
            z = x.view(1, x.shape[0],-1)
            if other_dct is not None:
                other_dct_lst = [other_dct[k].to(x.device).view(1, other_dct[k].shape[0], -1) for k in sorted(list(other_dct.keys()))]
                z = th.cat(
                [z] + other_dct_lst, -1)

        zshape = z.shape
        x = F.relu(self.fc0(z.view(-1, z.shape[-1]))).view(*zshape[:2], -1)
        hidden_lst = []
        if hidden is None:
            hidden = th.zeros_like(x[0]).squeeze(0)
        for _x in x:
            try:
                hidden = self.rnn_cell(_x, hidden)
            except Exception as e:
                b = 5
            hidden_lst.append(hidden)
        h = th.stack(hidden_lst)
        o = self.fc1(h.view(-1, h.shape[-1]))
        q = self.fc2(o)
        return {"out": q.view(zshape[0], zshape[1], -1) if is_sequence else q, "hidden": hidden}


class PongNet(th.nn.Module):

    def __init__(self, obs_space, n_actions=6):
        super().__init__()
        from stable_baselines3_lib.torch_layers import NatureCNN
        self.net = NatureCNN(obs_space)
        self.linear_q = th.nn.Linear(512, n_actions)

    def forward(self, input, other_dct, is_sequence=False):
        shp0 = input.shape[0]
        shp1 = input.shape[1]
        stacked_input = th.cat([input.float(), other_dct["log_beta"].unsqueeze(1).unsqueeze(1).repeat(1,84,84,1)], -1)
        out = self.net(stacked_input.permute(0,3,1,2))
        out = self.linear_q(out)
        return {"out": out if not is_sequence else out.view(shp0, shp1, *input.shape[-3:])}

class PongNet2(th.nn.Module):

    def __init__(self, obs_space, n_actions=6):
        super().__init__()
        from stable_baselines3_lib.torch_layers import NatureCNN
        self.net = NatureCNN(obs_space)
        self.linear_q = th.nn.Linear(512, 64)
        self.linear_q2 = th.nn.Linear(64, n_actions)

    def forward(self, input, other_dct, is_sequence=False):
        shp0 = input.shape[0]
        shp1 = input.shape[1]
        stacked_input = th.cat([input, other_dct["log_beta"].unsqueeze(1).unsqueeze(1).repeat(1,84,84,1)], -1)
        out = self.net(stacked_input.permute(0,3,1,2))
        out = F.relu(self.linear_q(out))
        out = self.linear_q2(out)
        return {"out": out if not is_sequence else out.view(shp0, shp1, *input.shape[-3:])}

class PongNet3(th.nn.Module):

    def __init__(self, obs_space, n_actions=6):
        super().__init__()
        from stable_baselines3_lib.torch_layers import NatureCNN
        import numpy as np
        import gym
        obs_space = gym.spaces.Box( np.zeros((4,84,84), dtype=np.float32) , np.zeros((4,84,84), dtype=np.float32)+255)
        self.net = NatureCNN(obs_space)
        self.linear_q = th.nn.Linear(512+1, 64)
        self.linear_q2 = th.nn.Linear(64, n_actions)

    def forward(self, input, other_dct, is_sequence=False):
        shp0 = input.shape[0]
        shp1 = input.shape[1]
        stacked_input = input.float()
        out = self.net(stacked_input.permute(0,3,1,2))
        out2 = th.cat([out, other_dct["log_beta"]], -1)
        out = F.relu(self.linear_q(out2))
        out = self.linear_q2(out)
        return {"out": out if not is_sequence else out.view(shp0, shp1, *input.shape[-3:])}

class BreakoutNet(th.nn.Module):

    def __init__(self, obs_space, n_actions=4):
        super().__init__()
        from stable_baselines3_lib.torch_layers import NatureCNN
        import numpy as np
        import gym
        obs_space = gym.spaces.Box( np.zeros((4,84,84), dtype=np.float32) , np.zeros((4,84,84), dtype=np.float32)+255)
        self.net = NatureCNN(obs_space)
        self.linear_q = th.nn.Linear(512+1, 64)
        self.linear_q2 = th.nn.Linear(64, n_actions)

    def forward(self, input, other_dct, is_sequence=False):
        shp0 = input.shape[0]
        shp1 = input.shape[1]
        stacked_input = input.float()
        out = self.net(stacked_input.permute(0,3,1,2))
        out2 = th.cat([out, other_dct["log_beta"]], -1)
        out = F.relu(self.linear_q(out2))
        out = self.linear_q2(out)
        return {"out": out if not is_sequence else out.view(shp0, shp1, *input.shape[-3:])}



class BreakoutSB3Net(th.nn.Module):

    def __init__(self, obs_space, n_actions=4):
        super().__init__()

        from rl_baselines3_zoo_lib import train
        expmanager = train.main(["--algo", "dqn", "--env",
                                 "BreakoutNoFrameskip-v4",
                                 "-i",
                                 "models/BreakoutNoFrameskip-v4/BreakoutNoFrameskip-v4.zip",
                                 "-n", "5000"])
        self.model = expmanager.setup_experiment().policy.q_net

    def forward(self, input, other_dct, is_sequence=False):
        out = self.model(input)
        return {"out": out if not is_sequence else out.view(shp0, shp1, *input.shape[-3:])}

class CartpoleSB3Net(th.nn.Module):

    def __init__(self, obs_space, n_actions=4, other_shape_dct=None, input_shape=None):
        super().__init__()

        from rl_baselines3_zoo_lib import train
        expmanager = train.main(["--algo", "dqn", "--env",
                                 "CartPole-v1",
                                 "-i",
                                 "models/CartPole-v1/CartPole-v1.zip",
                                 "-n", "5000"])
        self.model = expmanager.setup_experiment().policy.q_net

    def forward(self, input, other_dct, is_sequence=False):
        out = self.model(input)
        return {"out": out if not is_sequence else out.view(shp0, shp1, *input.shape[-3:])}

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

########################################################################################################################
# SET UP NETWORK REGISTRY
########################################################################################################################

network_registry = {"tab_grid": TabularGridNet,
                    "mlp_deep": MLPNetDeep,
                    "mlp_lstm_deep": MLPLSTMNetDeep,
                    "pong": PongNet,
                    "pong3": PongNet3,
                    "pongres": PongNet3,
                    "breakout": BreakoutNet,
                    "breakoutsb3": BreakoutSB3Net,
                    "cartpolesb3": CartpoleSB3Net}

def get_models(net_tag, n_actions, max_steps, obs_shape, other_shapes_dct, device):
    assert net_tag in network_registry, "Network {} not in registry!".format(net_tag)
    if net_tag in ["tab_grid"]:
        net = network_registry[net_tag](n_actions=n_actions,
                                        grid_shape=obs_shape,
                                        max_steps=max_steps).to(device)
        target_net = network_registry[net_tag](n_actions=n_actions,
                                                grid_shape=obs_shape,
                                                max_steps=max_steps,).to(device)
        return {"net": net, "target_net": target_net}, lambda: net.parameters()
    elif net_tag in ["pong"]:
        obs_space = th.load("models/pong_obsspace.pt")
        net = PongNet(obs_space).to(device)
        target_net = PongNet(obs_space).to(device)
        return {"net": net, "target_net": target_net}, lambda: net.parameters()
    elif net_tag in ["pong3"]:
        obs_space = th.load("models/pong_obsspace.pt")
        net = PongNet3(obs_space).to(device)
        target_net = PongNet3(obs_space).to(device)
        return {"net": net, "target_net": target_net}, lambda: net.parameters()
    elif net_tag in ["pongres"]:
        obs_space = th.load("models/pong_obsspace.pt")
        net = PongNet3(obs_space, n_actions=2).to(device)
        target_net = PongNet3(obs_space, n_actions=2).to(device)
        return {"net": net, "target_net": target_net}, lambda: net.parameters()
    elif net_tag in ["breakout"]:
        obs_space = th.load("models/pong_obsspace.pt")
        net = BreakoutNet(obs_space).to(device)
        target_net = BreakoutNet(obs_space).to(device)
        return {"net": net, "target_net": target_net}, lambda: net.parameters()
    elif net_tag in ["breakoutsb3"]:
        obs_space = th.load("models/pong_obsspace.pt")
        net = BreakoutSB3Net(obs_space).to(device)
        target_net = BreakoutSB3Net(obs_space).to(device)
        return {"net": net, "target_net": target_net}, lambda: net.parameters()
    elif net_tag in ["cartpolesb3"]:
        obs_space = th.load("models/pong_obsspace.pt")
        net = CartpoleSB3Net(obs_space).to(device)
        target_net = CartpoleSB3Net(obs_space).to(device)
        return {"net": net, "target_net": target_net}, lambda: net.parameters()
    else:
        net = network_registry[net_tag](n_actions=n_actions,
                                         input_shape=obs_shape,
                                         other_shape_dct=other_shapes_dct).to(device)
        target_net = network_registry[net_tag](n_actions=n_actions,
                                                input_shape=obs_shape,
                                                other_shape_dct=other_shapes_dct).to(device)
        return {"net": net, "target_net": target_net}, lambda: net.parameters()


