from stable_baselines3 import PPO
from network_env import SignalSyncEnv

env = SignalSyncEnv()

# Train the RL Agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./signalsync_tensorboard/")
model.learn(total_timesteps=100000)

# Save for GitHub deliverable
model.save("signalsync_ppo_agent")
