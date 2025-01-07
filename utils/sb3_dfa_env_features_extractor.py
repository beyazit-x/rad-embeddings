from model import Encoder
from utils.utils import feature_inds, dfa2feat
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DFAEnvFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, encoder_cls=Encoder, n_tokens=10):
        super().__init__(observation_space, features_dim)
        in_feat_size = n_tokens + len(feature_inds)
        self.encoder = encoder_cls(in_feat_size, features_dim)
        self.n_tokens = n_tokens

    def forward(self, dfa):
        feat = dfa2feat(dfa, n_tokens=self.n_tokens)
        return self.encoder(feat)
