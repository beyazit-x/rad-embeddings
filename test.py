import torch
import numpy as np
from dfa_gym import DFAEnv
from stable_baselines3 import PPO
from utils import DFAFeaturesExtractor, dfa2dist
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

env = DFAEnv()
check_env(env)

model = PPO.load("model.zip")
model.set_parameters("model.zip")

for param in model.policy.parameters():
    param.requires_grad = False

print("Total number of parameters:", sum(p.numel() for p in model.policy.parameters() if p.requires_grad))
print(model.policy)


n = 100
success_count = 0
a = []
b = []
for _ in range(n):
    obs, _ = env.reset()
    done = False
    while not done:
        features = model.policy.features_extractor(obs)
        state_value = model.policy.value_net(features)
        dist = model.policy.get_distribution(torch.from_numpy(obs))
        logits = dist.distribution.logits

        a.append(state_value.item())
        b.append(dfa2dist(obs))

        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(int(action))
        if reward > 0:
            success_count += 1

        # a.append(dist.distribution.probs[0][action.item()].item())
        # print(dist.distribution.probs)
        # print(torch.log(dist.distribution.probs))
        # print(torch.log(dist.distribution.probs) + state_value)
        # input()
        # print(action, dist.distribution.probs, state_value)
        # input()
        # print(state_value, reward, done, dfa2dist(obs))
        # input()

print(success_count/n)

correlation_matrix = np.corrcoef(a, b)
correlation_coefficient = correlation_matrix[0, 1]

print("Pearson correlation coefficient:", correlation_coefficient)


