from dfa_gym import DFAEnv, DFAWrapper
import token_env
from stable_baselines3 import PPO
from utils import TokenEnvFeaturesExtractor, LoggerCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from dfa_samplers import ReachAvoidSampler

import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 10_000_000,
    "env_name": "DFAWrapper<TokenEnv-v0>",
}
run = wandb.init(
    project="token_env_fixed_init_reach_avoid",
    id="rad_4",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

n_envs = 16
env_kwargs = {"env_id": "TokenEnv-v0", "label_f": token_env.TokenEnv.label_f}
env = make_vec_env(DFAWrapper, env_kwargs=env_kwargs, n_envs=n_envs)

policy_kwargs = dict(
    features_extractor_class=TokenEnvFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=1056),
    net_arch=dict(pi=[64, 64, 64], vf=[64, 64]),
    share_features_extractor=True,
)

gamma = 0.94
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=gamma,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=1e-2,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=10,
    tensorboard_log=f"token_env_policy/runs/{run.id}"
    )


print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)

model.learn(config["total_timesteps"], callback=[LoggerCallback(gamma=gamma), WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"token_env_policy/models/{run.id}",
        verbose=2,
    )]
)

model.save("token_env_policy/token_env_policy")

run.finish()
