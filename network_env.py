import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SignalSyncEnv(gym.Env):
    def __init__(self):
        super(SignalSyncEnv, self).__init__()
        # State S: [RTT_ms, Jitter_ms, Packet_Loss_Rate, Buffer_Fill, Render_FPS]
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(5,), dtype=np.float32)
        
        # Action A: [FEC_Ratio (0.0 to 0.5), Pacing_Delay_ms (0.0 to 100.0)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]), 
            high=np.array([0.5, 100.0]), 
            dtype=np.float32
        )
        
    def step(self, action):
        fec_ratio, pacing_delay = action
        
        # 1. Math: Update queue state based on incoming bandwidth and pacing delay
        # 2. Math: Calculate next observation (S')
        # 3. Math: Calculate reward (R)
        
        reward = -next_rtt - (50 * next_loss_rate) - (10 * fec_ratio)
        terminated = False # Continuous episode
        truncated = False
        
        return next_observation, reward, terminated, truncated, {}
        
    def reset(self, seed=None):
        # Reset queue to 0 and RTT to baseline (e.g., 20ms)
        return initial_observation, {}
