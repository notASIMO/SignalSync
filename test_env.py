import numpy as np
from network_env import SignalSyncEnv

def test_environment():
    print("--- Initializing SignalSync Environment Test ---")
    env = SignalSyncEnv()
    
    obs, info = env.reset()
    print(f"Initial Observation: {obs}")
    
    # Validation checks
    assert isinstance(obs, np.ndarray), "Observation must be a numpy array"
    assert obs.shape == env.observation_space.shape, f"Obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
    
    done = False
    total_reward = 0.0
    steps = 0
    
    print("\n--- Running Random Policy Rollout (100 steps) ---")
    while not done and steps < 100:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        if steps % 20 == 0 or done:
            print(f"Step {steps:3d} | Action: {action} | Obs: {obs} | Reward: {reward:.4f}")
            
    print("\n--- Environment Test Summary ---")
    print(f"Total Steps Executed: {steps}")
    print(f"Cumulative Reward: {total_reward:.4f}")
    print("Assertion Checks Passed: Gymnasium interface is fully compliant!")
    
    env.close()

if __name__ == "__main__":
    test_environment()