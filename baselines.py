import os
import numpy as np
from stable_baselines3 import PPO
from network_env import SignalSyncEnv as NetworkEnv

class FixedPacingBaseline:
    """Baseline agent that applies constant FEC redundancy and pacing delay."""
    def __init__(self, fec_rate=0.2, pacing_delay=5.0):
        self.fec_rate = fec_rate
        self.pacing_delay = pacing_delay

    def predict(self, observation):
        return NetworkEnv.encode_action(self.fec_rate, self.pacing_delay)

class AIMDBaseline:
    """Additive Increase Multiplicative Decrease baseline policy."""
    def __init__(self, min_delay=4.0, max_delay=12.0):
        self.current_delay = 6.0
        self.min_delay = min_delay
        self.max_delay = max_delay

    def predict(self, observation):
        # State S: [Recent_RTT (0), Jitter (1), Effective_Loss (2), Buffer_Fill (3), Render_FPS (4)]
        loss_rate = observation[2] if len(observation) > 2 else 0.0
        buffer_fill = observation[3] if len(observation) > 3 else 0.0
        
        # If high loss or buffer bloat, multiplicatively increase delay (back off)
        if loss_rate > 0.05 or buffer_fill > 25.0:
            self.current_delay = min(self.max_delay, self.current_delay * 1.25)
        else:  # Additive decrease of delay to improve throughput/FPS
            self.current_delay = max(self.min_delay, self.current_delay - 0.4)

        fec_rate = 0.1 if loss_rate < 0.02 else 0.35
        return NetworkEnv.encode_action(fec_rate, self.current_delay)

def evaluate_agent(agent, name="Baseline", num_episodes=20, is_sb3=False):
    env = NetworkEnv()
    rewards, rtts, losses, fps_list = [], [], [], []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            if is_sb3:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            rtts.append(obs[0])
            losses.append(obs[2] * 100.0)
            fps_list.append(obs[4])
            done = terminated or truncated
            
        rewards.append(episode_reward)
        
    print(f"[{name}] Evaluation over {num_episodes} episodes:")
    print(f"   Mean Episode Reward : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"   Avg RTT             : {np.mean(rtts):.2f} ms")
    print(f"   Avg Effective Loss  : {np.mean(losses):.2f} %")
    print(f"   Avg Render FPS      : {np.mean(fps_list):.2f} FPS\n")
    return np.mean(rewards), np.std(rewards)

if __name__ == "__main__":
    print("--- Evaluating SignalSync Baselines vs. RL Agent ---")
    fixed_agent = FixedPacingBaseline(fec_rate=0.15, pacing_delay=4.5)
    evaluate_agent(fixed_agent, name="Fixed Pacing (UDP-like)")

    aimd_agent = AIMDBaseline()
    evaluate_agent(aimd_agent, name="AIMD Control (TCP-like)")

    model_path = "models/ppo_signalsync.zip"
    if os.path.exists(model_path):
        ppo_model = PPO.load(model_path)
        evaluate_agent(ppo_model, name="PPO SignalSync Agent (RL)", is_sb3=True)