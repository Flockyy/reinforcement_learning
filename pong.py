import numpy as np
import time
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py

from collections import deque # FIFO queue data structurefrom tqdm import tqdm  # progress barsfrom gymnasium.wrappers import FrameStack
# from gymnasium.wrappers.atari_preprocessing
from stable_baselines3.common.atari_wrappers import (
  AtariWrapper,
  FireResetEnv,
)

gym.register_envs(ale_py) # we need to register ALE with gym

# use cuda if you have it otherwise cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')

observation, _ = env.reset()

for _ in range(100):  
  action = env.action_space.sample()
  observation_prime, reward, terminated, truncated, _ = env.step(action)
  observation = observation_prime
  
env = AtariWrapper(env, terminal_on_life_loss=False, frame_skip=4)

class LazyFramesToNumpyWrapper(gym.ObservationWrapper): # subclass obswrapper
    def __init__(self, env):
        super().__init__(env)
        self.env = env # the environment that we want to convert

    def observation(self, observation):
        # if its a LazyFrames object then turn it into a numpy array
        if isinstance(observation, LazyFrames):
            return np.array(observation)
        return observation

    def make_env(game, render='rgb_array'):
        env = gym.make(game, render_mode=render)
        env = AtariWrapper(env, terminal_on_life_loss=False, frame_skip=4)
        env = FrameStack(env, num_stack=4)
        env = LazyFramesToNumpyWrapper(env)
        # sometimes a environment needs that the fire button be
        # pressed to start the game, this makes sure that game is started when needed
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        return env