import torch
import numpy as np
from dfa import DFA
from model import Encoder
from dfa.utils import dfa2dict
from collections import OrderedDict
from torch_geometric.data import Data
from torch_geometric.data import Batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

feature_inds = {"temp": -5, "rejecting": -4, "accepting": -3, "init": -2, "normal": -1}

class DFAFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, encoder_cls=Encoder, n_tokens=10):
        super().__init__(observation_space, features_dim)
        in_feat_size = n_tokens + len(feature_inds)
        self.encoder = encoder_cls(in_feat_size, features_dim)

    def forward(self, dfa):
        feat = dfa2feat(dfa)
        return self.encoder(feat)

def dfa2feat(dfa_obs, n_tokens=10):
    return Batch.from_data_list(list(map(lambda x: _dfa2feat(x, n_tokens=n_tokens), dfa_obs)))

def _dfa2feat(dfa_obs, n_tokens=10):

    tokens = list(range(n_tokens))
    feature_size = len(tokens) + len(feature_inds)

    dfa_int = int("".join(map(str, map(int, dfa_obs.squeeze().tolist()))))
    dfa = DFA.from_int(dfa_int, tokens)

    dfa_dict, s_init = dfa2dict(dfa)

    nodes = OrderedDict({s: np.zeros(feature_size) for s in dfa_dict.keys()})
    edges = [(s, s) for s in nodes]
    
    for s in dfa_dict.keys():

        label, transitions = dfa_dict[s]
        leaving_transitions = [1 if s != transitions[a] else 0 for a in transitions.keys()]

        if s not in nodes:
            nodes[s] = np.zeros(feature_size)

        nodes[s][feature_inds["normal"]] = 1
        if s == s_init:
            nodes[s][feature_inds["init"]] = 1
        if label: # is accepting?
            nodes[s][feature_inds["accepting"]] = 1
        elif sum(leaving_transitions) == 0: # is rejecting?
            nodes[s][feature_inds["rejecting"]] = 1

        for e in dfa_dict.keys():
            if s == e:
                continue
            for a in transitions:
                if transitions[a] == e:
                    if (s, e) not in nodes:
                        nodes[(s, e)] = np.zeros(feature_size)
                        nodes[(s, e)][feature_inds["temp"]] = 1
                    nodes[(s, e)][a] = 1
                    s_idx = list(nodes.keys()).index(s)
                    t_idx = list(nodes.keys()).index((s, e))
                    e_idx = list(nodes.keys()).index(e)
                    # Reverse
                    if (e_idx, t_idx) not in edges:
                        edges.append((e_idx, t_idx))
                    if (t_idx, t_idx) not in edges:
                        edges.append((t_idx, t_idx))
                    if (t_idx, s_idx) not in edges:
                        edges.append((t_idx, s_idx))

    feat = torch.from_numpy(np.array(list(nodes.values())))
    edge_index = torch.from_numpy(np.array(edges))

    current_state = torch.from_numpy(np.array([1] + [0] * (len(nodes) - 1))) # 0 is the current state

    return Data(feat=feat, edge_index=edge_index.T, current_state=current_state)