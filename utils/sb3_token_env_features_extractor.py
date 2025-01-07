import torch
from torch import nn
from model import Encoder
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TokenEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, encoder_cls=Encoder, n_tokens=10):
        super().__init__(observation_space, features_dim)
        model = PPO.load("dfa_encoder/dfa_encoder")
        model.set_parameters("dfa_encoder/dfa_encoder")
        for param in model.policy.parameters():
            param.requires_grad = False
        self.encoder = model.policy
        self.encoder.eval()
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
        rad = self.encoder.features_extractor(dfa_obs)
        # logits = self.encoder.get_distribution(dfa_obs).distribution.logits
        # value = self.encoder.predict_values(dfa_obs)
        # q_values = logits + value
        # rad = q_values
        obs = self.image_conv(obs)
        obs = torch.cat((obs, rad), dim=1)
        return obs
