import torch
from torch import nn
from model import Encoder
from utils.utils import get_dfa_encoder
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TokenEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, encoder_cls=Encoder, n_tokens=10):
        super().__init__(observation_space, features_dim)
        self.dfa_encoder = get_dfa_encoder()
        c, w, h = observation_space["obs"].shape # CxWxH
        self.image_conv = nn.Sequential(
            nn.Conv2d(c, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, dict_obs):
        dfa_obs = dict_obs["dfa_obs"]
        obs = dict_obs["obs"]
        rad = self.dfa_encoder(dfa_obs)
        obs = self.image_conv(obs)
        obs = torch.cat((obs, rad), dim=1)
        return obs
