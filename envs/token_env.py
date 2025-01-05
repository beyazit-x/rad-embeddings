import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TokenEnv(gym.Env):
    def __init__(self, n_tokens=10, size=(7, 7), timeout=75):
        super().__init__()
        assert size[0] % 2 == 1 and size[1] % 2 == 1
        
        self.n_tokens = n_tokens
        self.size = size
        self.timeout = timeout

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_tokens, *self.size), dtype=np.uint8)

        self.t = 0
        
        self.agent = None
        x = np.arange(self.size[0])
        y = np.arange(self.size[1])
        xx, yy = np.meshgrid(x, y)
        self.grid = np.column_stack((xx.ravel(), yy.ravel()))
        self.grid_size = self.size[0] * self.size[1]

        self.action_map = {0: (0, 1),
                           1: (1, 0),
                           2: (0, -1),
                           3: (-1, 0)}


    def reset(self, seed=None):

        np.random.seed(seed)

        indices = np.random.choice(self.grid_size, 2 * self.n_tokens + 1, replace=False)
        samples = self.grid[indices]

        self.agent = samples[0]
        
        samples = samples[1:]
        np.random.shuffle(samples)
        self.tokens = [(i % self.n_tokens, samples[i]) for i in range(2 * self.n_tokens)]

        self.t = 0

        obs = self._get_obs()

        return obs, {}


    def step(self, action):

        dx, dy = self.action_map[action]
        agent_x = (self.agent[0] + dx + self.size[0]) % self.size[0]
        agent_y = (self.agent[1] + dy + self.size[1]) % self.size[1]
        self.agent = (agent_x, agent_y)

        obs = self._get_obs()

        reward = 0

        self.t += 1
        done = self.t >= self.timeout

        return obs, reward, done, False, {}


    def _get_obs(self):
        center_x = self.size[0] // 2
        center_y = self.size[1] // 2
        delta  = center_x - self.agent[0], center_y - self.agent[1]

        obs = np.zeros(shape=(self.n_tokens, *self.size), dtype=np.uint8)

        for i, xy in self.tokens:
            rel_xy = (xy + delta + self.size) % self.size
            obs[i, *rel_xy] = 1
        
        return obs


    @staticmethod
    def label_f(obs):
        # token, _, _ = np.bitwise_and(obs[:-1], obs[-1]).nonzero()
        # if token.size > 1:
        #     raise RuntimeError("At most one token can be in a single cell.")
        # elif token.size == 1:
        #     return token.item()
        # return None
        token = np.where(obs[:, obs.shape[1] // 2, obs.shape[2] // 2] == 1)[0]
        if token.size > 1:
            raise RuntimeError("At most one token can be in a single cell.")
        elif token.size == 1:
            return token.item()
        return None


if __name__ == '__main__':
    # env = TokenEnv(n_tokens=3, size=(3, 3), token_ratio=0.5, timeout=2)
    env = TokenEnv()
    obs, info = env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        action = env.action_space.sample()
        print(i, obs, action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        print("Label:", env.label_f(obs))
        done = terminated or truncated
    env.close()

    # from dfa_gym import DFAWrapper
    # from dfa_samplers import ReachAvoidSampler
    # from stable_baselines3.common.env_util import make_vec_env

    # env = DFAWrapper(TokenEnv(), sampler=ReachAvoidSampler())
    # wrapper_class = DFAWrapper
    # wrapper_kwargs = {"label_f": TokenEnv.label_f}

    # n_envs = 1
    # env = make_vec_env(TokenEnv, wrapper_class=wrapper_class, wrapper_kwargs=wrapper_kwargs, n_envs=n_envs)
    # env_kwargs = {"env": TokenEnv,
    #         "label_f": TokenEnv.label_f}

    # n_envs = 16
    # env = make_vec_env(DFAWrapper, env_kwargs=env_kwargs, n_envs=n_envs)
    # total_reward = 0
    # n = 1000
    # for _ in range(n):
    #     i = 0
    #     done = False
    #     obs, info = env.reset()
    #     # print("Hey")
    #     while not done:
    #         i += 1
    #         action = [env.action_space.sample() for _ in range(n_envs)]
    #         obs, reward, terminated, truncated = env.step(action)
    #         print(obs["obs"].shape, obs["dfa_obs"].shape, reward)
    #         input()
    #         total_reward += reward.sum()
    #         done = terminated.any()
    #     env.close()
    # print(total_reward/n)

