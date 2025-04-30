import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, GrayscaleObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import ale_py

gym.register_envs(ale_py)

# Load the arcade Mario Bros. environment
env = gym.make("ALE/MarioBros-v5", render_mode="rgb_array")

# Wrap the environment for better performance in training
env = MaxAndSkipEnv(env, skip=4)
env = GrayscaleObservation(env, keep_dim=True)
# env = ResizeObservation(env, shape=84)
# env = FrameStackObservation(env, num_stack=4)


from stable_baselines3 import DQN

model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1
)

model.learn(total_timesteps=200_000)
model.save("dqn_mario_arcade")

import time

model = DQN.load("dqn_mario_arcade", env=env)

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    time.sleep(0.01)  # Slow down rendering