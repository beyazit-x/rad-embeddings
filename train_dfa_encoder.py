import dfa_gym
from stable_baselines3 import PPO
from utils import DFAEnvFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env


env = make_vec_env("DFAEnv-v0", n_envs=16)


policy_kwargs = dict(
    features_extractor_class=DFAEnvFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=32),
    net_arch=dict(pi=[], vf=[]),
    share_features_extractor=True,
)


model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    n_steps=512,
    batch_size=1024,
    n_epochs=2,
    gamma=0.9,
    gae_lambda=0.5,
    clip_range=0.1,
    ent_coef=1e-2,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=10,
    tensorboard_log="dfa_encoder"
    )


print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)
model.learn(1_000_000)
model.save("dfa_encoder/dfa_encoder")
