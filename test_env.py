import numpy as np
 
from network_env import SignalSyncEnv, NetworkChannel
 
try:
    from gymnasium.utils.env_checker import check_env
except ImportError:
    check_env = None
 
 
# standard gymnasium API check
def test_gymnasium_interface():
    if check_env is None:
        print("SKIPPED: gymnasium check_env not available")
        return
    check_env(SignalSyncEnv())
 
 
# random actions should never produce NaN/inf or leave the observation space
def test_10000_random_steps_finite_and_in_bounds():
    env = SignalSyncEnv()
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, np.ndarray) and obs.shape == env.observation_space.shape
    truncated_seen = False
    for _ in range(10_000):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)), f"non-finite observation: {obs}"
        assert np.isfinite(reward), f"non-finite reward: {reward}"
        assert env.observation_space.contains(obs), f"observation out of bounds: {obs}"
        if terminated or truncated:
            truncated_seen |= bool(truncated)
            obs, _ = env.reset()
    assert truncated_seen, "episodes never ended (truncation path untested)"
 
 
# the env must use its own seeded RNG, not the global numpy one
def test_same_seed_is_reproducible_even_if_global_rng_differs():
    def rollout(seed, global_seed):
        np.random.seed(global_seed)
        env = SignalSyncEnv()
        env.reset(seed=seed)
        action = np.array([0.0, 0.0], dtype=np.float32)
        return [env.step(action)[1] for _ in range(60)]
 
    assert rollout(1, 111) == rollout(1, 999)
    assert rollout(1, 0) != rollout(2, 0)
 
 
# policies sending different packet counts must still see the same Good/Bad sequence
def test_every_policy_faces_the_same_network():
    a, b = SignalSyncEnv(), SignalSyncEnv()
    a.reset(seed=7)
    b.reset(seed=7)
    act_a = SignalSyncEnv.encode_action(0.15, 6.5)
    act_b = SignalSyncEnv.encode_action(0.10, 4.2)
    for _ in range(150):
        a.step(act_a)
        b.step(act_b)
        assert a.channel.current_state == b.channel.current_state
 
 
# regression test: the channel state used to reset every packet, so bursts had no memory
def test_congestion_bursts_persist_across_ticks():
    env = SignalSyncEnv(max_steps=10**9)
    env.reset(seed=3)
    action = np.array([0.0, 0.0], dtype=np.float32)
    states = []
    for _ in range(4000):
        env.step(action)
        states.append(env.channel.current_state)
    s = np.array(states, dtype=float)
    lag1 = np.corrcoef(s[:-1], s[1:])[0, 1]
    assert lag1 > 0.5, f"channel has no memory across ticks (lag-1 correlation {lag1:.3f})"
    assert 0.1 < s.mean() < 0.3, f"unexpected share of Bad ticks: {s.mean():.2f}"
 
 
# encode/decode should be inverses and hit the documented ranges
def test_action_decoding_ranges():
    lo = SignalSyncEnv.decode_action(np.array([-1.0, -1.0]))
    hi = SignalSyncEnv.decode_action(np.array([1.0, 1.0]))
    assert np.allclose(lo, (0.0, 3.5)) and np.allclose(hi, (0.5, 14.0))
    for fec, delay in [(0.15, 4.5), (0.3, 9.0)]:
        back = SignalSyncEnv.decode_action(SignalSyncEnv.encode_action(fec, delay))
        assert np.allclose(back, (fec, delay), atol=1e-5)
 
 
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print("\nAll checks passed.")