import argparse
import os
import shutil
import warnings
 
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
 
from baselines import MODEL_PATH, OBS_SCALE, VAL_SEEDS, ScaledPolicy, evaluate_policy
from network_env import SignalSyncEnv
 
warnings.filterwarnings("ignore")
 
REWARD_SCALE = 0.1  # only PPO sees scaled rewards
 
 
class ScaleObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 20.0, shape=(5,), dtype=np.float32)
 
    def observation(self, obs):
        return (np.asarray(obs, dtype=np.float32) / OBS_SCALE).astype(np.float32)
 
 
class ScaleReward(gym.RewardWrapper):
    def reward(self, reward):
        return reward * REWARD_SCALE
 
 
def make_env(seed):
    # Monitor goes inside the wrappers so the learning curve logs raw rewards
    env = Monitor(SignalSyncEnv(), f"results/train_monitor_seed{seed}")
    return ScaleReward(ScaleObs(env))
 
 
def train_one(seed, timesteps):
    set_random_seed(seed)
    model = PPO(
        "MlpPolicy", make_env(seed), seed=seed, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0),
        tensorboard_log="./logs/",
    )
    model.learn(total_timesteps=timesteps)
    path = f"models/ppo_signalsync_seed{seed}"
    model.save(path)
    return model, path + ".zip"
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()
 
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
 
    # pick the best seed on validation episodes, not the test episodes used in the report
    scores = {}
    for seed in args.seeds:
        print(f"\n=== Training PPO, seed {seed}, {args.timesteps} steps ===")
        model, path = train_one(seed, args.timesteps)
        res = evaluate_policy(ScaledPolicy(model), is_sb3=True, seeds=VAL_SEEDS)
        scores[seed] = (res["reward"][0], path)
        print(f"Seed {seed}: validation reward {res['reward'][0]:.1f}")
 
    best_seed = max(scores, key=lambda s: scores[s][0])
    shutil.copyfile(scores[best_seed][1], MODEL_PATH)
    print("\nValidation rewards per seed:", {s: round(v[0], 1) for s, v in scores.items()})
    print(f"Best seed {best_seed} saved as {MODEL_PATH}")
 
 
if __name__ == "__main__":
    main()