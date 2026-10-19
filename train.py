import os
import warnings
from stable_baselines3 import PPO
from network_env import SignalSyncEnv

warnings.filterwarnings("ignore")

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Initializing SignalSync environment...")
    env = SignalSyncEnv()

    print("Building PPO Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./logs/"
    )

    print("Training PPO Agent for 50,000 steps...")
    model.learn(total_timesteps=50000)

    model_path = "models/ppo_signalsync"
    model.save(model_path)
    print(f"Model saved successfully to {model_path}.zip")

if __name__ == "__main__":
    train()