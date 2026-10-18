import numpy as np
from network_env import NetworkEnv

def test_environment():
    env = NetworkEnv()
    obs, info = env.reset()
    print("Initial Observation:", obs)
    
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done and steps < 100:
        action = env.action_space.sample()  # Random action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
    print(f"Test completed in {steps} steps. Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    test_environment()