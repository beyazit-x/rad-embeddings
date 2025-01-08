import wandb
import dfa_gym
import gymnasium as gym
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from utils import DFAEnvFeaturesExtractor, LoggerCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

run = wandb.init(project="sb3", sync_tensorboard=True)

n_envs = 16
env_id = "DFAEnv-v0"

env = gym.make(env_id)
check_env(env)
n_tokens = env.unwrapped.sampler.n_tokens

env = make_vec_env(env_id, n_envs=n_envs)

config = dict(
    policy = "MlpPolicy",
    env = env,
    learning_rate = 1e-3,
    n_steps = 512,
    batch_size = 1024,
    n_epochs = 2,
    gamma = 0.9,
    gae_lambda = 0.5,
    clip_range = 0.1,
    ent_coef = 1e-2,
    vf_coef = 0.5,
    max_grad_norm = 0.5,
    policy_kwargs = dict(
        features_extractor_class = DFAEnvFeaturesExtractor,
        features_extractor_kwargs = dict(features_dim=32, n_tokens=n_tokens),
        net_arch=dict(pi=[], vf=[]),
        share_features_extractor=True,
    ),
    verbose = 10,
    tensorboard_log = f"dfa_encoder/runs/{run.id}"
)

model = PPO(**config)

print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)

logger_callback = LoggerCallback(gamma=config["gamma"])
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_freq=100,
    model_save_path=f"dfa_encoder/models/{run.id}",
    verbose=config["verbose"])

model.learn(1_000_000, callback=[logger_callback, wandb_callback])
model.save("dfa_encoder/dfa_encoder")

run.finish()
