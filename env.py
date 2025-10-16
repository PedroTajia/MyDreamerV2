# import dm_control
import numpy as np
import imageio
import gymnasium as gym
from torchvision.transforms import Resize
import torch

import robosuite as suite
from robosuite.utils.log_utils import DefaultLogger
# Only show warnings/errors in console
_ = DefaultLogger(console_logging_level="WARNING").get_logger()

class RobotSuiteEnv(gym.Env):
    def __init__(self, env_name, robot, type_obs=torch.bfloat16, output_obs=(1, 256, 256), obs_resize=True, reward_shaping=True):
        super().__init__()
        self.c, self.h, self.w = output_obs
        self.env_name = env_name
        self.robot = robot
        self.env = suite.make(env_name, robots=self.robot, reward_shaping=reward_shaping)
        self.action_space = gym.spaces.Box(-1, 1, shape=(self.env.action_dim,))
        self.observation_space = gym.spaces.MultiBinary((self.c, self.h, self.w))
        self.obs_resize = obs_resize
        self.type_obs = type_obs
    
    def reset(self):
        obs = self.env.reset()
        self.obs = obs["agentview_image"] if "agentview_image" in obs else obs
        self.renderObs = self.obs
        if self.obs_resize:
           resize = Resize((self.h, self.w))
           self.obs = torch.tensor(self.obs.transpose(2, 0, 1), dtype=torch.bfloat16) / 255.0
           self.obs = resize(self.obs)
        return self.obs.flip(1).to(self.type_obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs = obs["agentview_image"] if "agentview_image" in obs else obs
        self.renderObs = self.obs
        if self.obs_resize:
           resize = Resize((self.h, self.w))
           self.obs = torch.tensor(self.obs.transpose(2, 0, 1), dtype=torch.bfloat16) / 255.0
           self.obs = resize(self.obs)
        
        self.terminal = done
        return self.obs.flip(1).to(self.type_obs), reward, done, {}
    
    def seed(self, seed='None'):
        self.env = suite.make(self.env_name, robot=self.robot, seed=seed)
        
    def render(self):
        return np.flip(self.renderObs)

