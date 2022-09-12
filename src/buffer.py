import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence

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

        assert len(set([len(self.buffer_content[k]) for k in self.buffer_content.keys()])) == 1, "buffer inconsistency!"
        for k in self.buffer_content.keys():
            if len(self.buffer_content[k]) + 1 >= self.buffer_len:
                self.buffer_content[k].pop(0)

        for k in self.buffer_scheme.keys():
            v = dct[k]
            self.buffer_content[k].append(v.to(self.buffer_device))

        pass

    def sample(self, sample_size, mode="transitions"):

        if mode in ["transitions"]:
            sz = self.size(mode="transitions")
            sample_ids = np.random.randint(1, sz + 1, size=sample_size)
            cs = np.cumsum([0] + [min(a.shape[0],
                                      o.shape[0]) for o, a in zip(self.buffer_content["obs"],
                                                                  self.buffer_content["action"])])
            main_idxs = np.searchsorted(cs, sample_ids) - 1
            sub_idxs = [sid - cs[mid] - 1 for mid, sid in zip(main_idxs, sample_ids)]
            ret_dict = {}
            for k in self.buffer_scheme.keys():
                ret_dict[k] = th.stack(
                    [self.buffer_content[k][mid][sid, ...] for mid, sid in zip(main_idxs, sub_idxs)])
            ret_dict["next_obs"] =  th.stack(
                    [self.buffer_content["obs"][mid][sid+1, ...] for mid, sid in zip(main_idxs, sub_idxs)])
            ret_dict["next_avail_actions"] =  th.stack(
                    [self.buffer_content["avail_actions"][mid][sid+1, ...] if \
                         self.buffer_content["avail_actions"][mid].shape[0] > sid+1 else self.buffer_content["avail_actions"][mid][0].clone().zero_()\
                     for mid, sid in zip(main_idxs, sub_idxs)])
            ret_dict["next_obs_is_terminal"] = th.stack([th.zeros(1,) + 1 \
                                                            if self.buffer_content["obs"][mid].shape[0] - 1 == sid + 1\
                                                            else th.zeros(1,) \
                    for mid, sid in zip(main_idxs, sub_idxs)])

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
            return sum([min(a.shape[0],
                        o.shape[0]) for o, a in zip(self.buffer_content["obs"],
                                                    self.buffer_content["action"])])
        else:
            raise Exception("Unknown size mode: {}".format(mode))