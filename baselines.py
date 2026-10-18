import numpy as np
from network_env import NetworkEnv

class FixedPacingBaseline:
    """Baseline agent that applies constant FEC redundancy and pacing delay."""
    def __init__(self, fec_rate=0.2, pacing_delay=5.0):
        self.fec_rate = fec_rate
        self.pacing_delay = pacing_delay

    def predict(self, observation):
        # Maps fixed parameters to normalized action space
        return np.array([self.fec_rate, self.pacing_delay], dtype=np.float32)

def evaluate_baseline(agent, num_episodes=10):
    env = NetworkEnv()
    rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        rewards.append(episode_reward)
        
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Baseline Evaluation over {num_episodes} episodes:")
    print(f"Mean Reward: {mean_reward:.4f} +/- {std_reward:.4f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    baseline = FixedPacingBaseline()
    evaluate_baseline(baseline)