from dfa_gym import DFAEnv, DFAWrapper
import token_env
from stable_baselines3 import PPO
from utils import TokenEnvFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from dfa_samplers import ReachAvoidSampler

n_envs = 16
env_kwargs = {"env_id": "TokenEnv-v0", "label_f": token_env.TokenEnv.label_f}
env = make_vec_env(DFAWrapper, env_kwargs=env_kwargs, n_envs=n_envs)


policy_kwargs = dict(
    features_extractor_class=TokenEnvFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=1056),
    net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64]),
    share_features_extractor=True,
)


model = PPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.94,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=1e-2,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=10,
    tensorboard_log="token_env_policy"
    )


print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)
model.learn(10_000_000)
model.save("token_env/token_env_policy")

