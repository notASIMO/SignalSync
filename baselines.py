import numpy as np
from network_env import SignalSyncEnv as NetworkEnv

class FixedPacingBaseline:
    """Baseline agent that applies constant FEC redundancy and pacing delay."""
    def __init__(self, fec_rate=0.2, pacing_delay=5.0):
        self.fec_rate = fec_rate
        self.pacing_delay = pacing_delay

    def predict(self, observation):
        return np.array([self.fec_rate, self.pacing_delay], dtype=np.float32)

class AIMDBaseline:
    """Additive Increase Multiplicative Decrease baseline policy."""
    def __init__(self, min_delay=1.0, max_delay=10.0):
        self.current_delay = min_delay
        self.min_delay = min_delay
        self.max_delay = max_delay

    def predict(self, observation):
        # observation standard index assumption: [rtt, loss_rate, buffer_occupancy, ...]
        loss_rate = observation[1] if len(observation) > 1 else 0.0
        
        # If high loss, multiplicatively increase delay (back off)
        if loss_rate > 0.05:
            self.current_delay = min(self.max_delay, self.current_delay * 1.5)
        else: # Additive decrease of delay to reduce latency
            self.current_delay = max(self.min_delay, self.current_delay - 0.5)

        fec_rate = 0.1 if loss_rate < 0.02 else 0.3
        return np.array([fec_rate, self.current_delay], dtype=np.float32)

def evaluate_baseline(agent, name="Baseline", num_episodes=10):
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
    print(f"[{name}] Evaluation over {num_episodes} episodes:")
    print(f"   Mean Reward: {mean_reward:.4f} +/- {std_reward:.4f}\n")
    return mean_reward, std_reward

if __name__ == "__main__":
    print("--- Evaluating Baselines ---")
    fixed_agent = FixedPacingBaseline()
    evaluate_baseline(fixed_agent, name="Fixed Pacing")

    aimd_agent = AIMDBaseline()
    evaluate_baseline(aimd_agent, name="AIMD Control")