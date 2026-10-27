import gymnasium as gym
from gymnasium import spaces
import numpy as np
 
 
class NetworkChannel:
    def __init__(self, rng=None):
        # transition probs are per tick, so a burst lasts ~5 ticks on average
        self.p_gb = 0.05
        self.p_bg = 0.20
        self.loss_prob_good = 0.01
        self.loss_prob_bad = 0.35
        self.current_state = 0
        self.base_rtt = 20.0
        self.rng = rng if rng is not None else np.random.default_rng()
 
    def advance_state(self):
        if self.current_state == 0 and self.rng.random() < self.p_gb:
            self.current_state = 1
        elif self.current_state == 1 and self.rng.random() < self.p_bg:
            self.current_state = 0
 
    def sample_packet(self):
        if self.current_state == 0:
            is_dropped = self.rng.random() < self.loss_prob_good
            rtt = self.base_rtt + self.rng.exponential(scale=2.0)
        else:
            is_dropped = self.rng.random() < self.loss_prob_bad
            rtt = self.base_rtt + self.rng.exponential(scale=35.0)
        return bool(is_dropped), float(rtt)
 
 
class SignalSyncEnv(gym.Env):
    def __init__(self, max_steps=200):
        super().__init__()
 
        # obs: [rtt, rtt change, effective loss, buffer fill, fps]
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.full(5, 1000.0, dtype=np.float32),
            dtype=np.float32,
        )
 
        # action in [-1, 1]^2 -> fec in [0, 0.5], pacing delay in [3.5, 14] ms
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
 
        self.tick_duration_ms = 100.0
        self.max_buffer_size = 100.0
        self.link_capacity = 20.0
        self.target_base_packets = 16.0
        self.max_steps = max_steps
 
        self.channel = NetworkChannel()
        self._episode_seed = 0
 
        self.buffer_fill = 0.0
        self.recent_rtt = 20.0
        self.recent_jitter = 0.0
        self.render_fps = 60.0
        self.current_step = 0
 
    @staticmethod
    def encode_action(fec_ratio, pacing_delay):
        norm_fec = np.clip((fec_ratio / 0.5) * 2.0 - 1.0, -1.0, 1.0)
        norm_delay = np.clip(((pacing_delay - 3.5) / 10.5) * 2.0 - 1.0, -1.0, 1.0)
        return np.array([norm_fec, norm_delay], dtype=np.float32)
 
    @staticmethod
    def decode_action(action):
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))
        fec_ratio = (a0 + 1.0) * 0.25
        pacing_delay = 3.5 + (a1 + 1.0) * 5.25
        return fec_ratio, pacing_delay
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # same reset seed -> same network for every policy
        self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.buffer_fill = 0.0
        self.recent_rtt = 20.0
        self.recent_jitter = 0.0
        self.render_fps = 60.0
        self.current_step = 0
        self.channel.current_state = 0
 
        initial_state = np.array(
            [self.recent_rtt, self.recent_jitter, 0.0, self.buffer_fill, self.render_fps],
            dtype=np.float32,
        )
        return initial_state, {}
 
    def step(self, action):
        self.current_step += 1
        fec_ratio, pacing_delay = self.decode_action(action)
 
        # channel randomness depends only on (episode seed, tick), not on how many packets we send
        self.channel.rng = np.random.default_rng([self._episode_seed, self.current_step])
        self.channel.advance_state()
 
        base_packets = self.tick_duration_ms / pacing_delay
        total_incoming = base_packets * (1.0 + fec_ratio)
 
        self.buffer_fill += total_incoming
        buffer_overflow_drops = max(0.0, self.buffer_fill - self.max_buffer_size)
        self.buffer_fill = min(self.buffer_fill, self.max_buffer_size)
 
        transmitted_packets = min(self.buffer_fill, self.link_capacity)
        self.buffer_fill -= transmitted_packets
 
        queueing_delay = self.buffer_fill * 2.0
 
        network_drops = 0
        rtt_samples = []
        for _ in range(int(round(transmitted_packets))):
            dropped, rtt = self.channel.sample_packet()
            if dropped:
                network_drops += 1
            else:
                rtt_samples.append(rtt)
 
        # fec recovers up to this many dropped packets
        fec_shield_packets = transmitted_packets * (fec_ratio / (1.0 + fec_ratio))
        unrecovered_network_drops = max(0.0, float(network_drops) - fec_shield_packets)
        total_unrecovered = unrecovered_network_drops + buffer_overflow_drops
        effective_loss = float(np.clip(
            total_unrecovered / max(transmitted_packets + buffer_overflow_drops, 1.0), 0.0, 1.0))
 
        delivered_rate_ratio = min(1.0, base_packets / self.target_base_packets) * (1.0 - effective_loss)
        self.render_fps = float(np.clip(60.0 * delivered_rate_ratio, 5.0, 60.0))
 
        if rtt_samples:
            new_rtt = float(np.mean(rtt_samples)) + queueing_delay
            self.recent_jitter = abs(new_rtt - self.recent_rtt)
            self.recent_rtt = new_rtt
        else:
            self.recent_rtt = 300.0
            self.recent_jitter = 80.0
 
        # reward is the negative sum of these penalties
        latency_penalty = 0.02 * self.recent_rtt + 0.01 * self.recent_jitter
        loss_penalty = 8.0 * effective_loss
        fps_penalty = 0.08 * (60.0 - self.render_fps)
        fec_overhead_penalty = 0.8 * fec_ratio
        reward = -(latency_penalty + loss_penalty + fps_penalty + fec_overhead_penalty)
 
        next_state = np.array([
            self.recent_rtt,
            self.recent_jitter,
            effective_loss,
            self.buffer_fill,
            self.render_fps,
        ], dtype=np.float32)
 
        truncated = self.current_step >= self.max_steps
        return next_state, float(reward), False, truncated, {}