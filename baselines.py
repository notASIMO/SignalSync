import hashlib
import json
import os
 
import numpy as np
 
from network_env import SignalSyncEnv as NetworkEnv
 
NAIVE_FEC, NAIVE_DELAY = 0.15, 4.5
# separate seed sets so tuning and model selection never touch the test episodes
EVAL_SEEDS = list(range(100, 200))
TUNE_SEEDS = list(range(1000, 1008))
VAL_SEEDS = list(range(2000, 2030))
MODEL_PATH = "models/ppo_signalsync.zip"
TUNED_CACHE = "results/tuned_fixed.json"
# PPO sees obs / OBS_SCALE so inputs are roughly 0-3
OBS_SCALE = np.array([100.0, 50.0, 1.0, 100.0, 60.0], dtype=np.float32)
 
 
class ScaledPolicy:
    def __init__(self, model):
        self.model = model
 
    def predict(self, obs, deterministic=True):
        return self.model.predict(np.asarray(obs, dtype=np.float32) / OBS_SCALE,
                                  deterministic=deterministic)
 
 
class FixedPacingBaseline:
    def __init__(self, fec_rate=NAIVE_FEC, pacing_delay=NAIVE_DELAY):
        self.fec_rate = fec_rate
        self.pacing_delay = pacing_delay
 
    def reset(self):
        pass
 
    def predict(self, observation):
        return NetworkEnv.encode_action(self.fec_rate, self.pacing_delay)
 
 
# back off (longer delay) on loss or a filling buffer, speed up slowly otherwise
class AIMDBaseline:
    def __init__(self, min_delay=4.0, max_delay=12.0, start_delay=6.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.start_delay = start_delay
        self.current_delay = start_delay
 
    def reset(self):
        self.current_delay = self.start_delay
 
    def predict(self, observation):
        loss_rate = observation[2]
        buffer_fill = observation[3]
        if loss_rate > 0.05 or buffer_fill > 25.0:
            self.current_delay = min(self.max_delay, self.current_delay * 1.25)
        else:
            self.current_delay = max(self.min_delay, self.current_delay - 0.4)
        fec_rate = 0.1 if loss_rate < 0.02 else 0.35
        return NetworkEnv.encode_action(fec_rate, self.current_delay)
 
 
def load_ppo(path=MODEL_PATH):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found - PPO is SKIPPED. Run `python train.py` first.")
        return None
    from stable_baselines3 import PPO
    return ScaledPolicy(PPO.load(path))
 
 
def run_episode(env, agent, seed, is_sb3=False):
    obs, _ = env.reset(seed=seed)
    if not is_sb3 and hasattr(agent, "reset"):
        agent.reset()
    done, total, rtts, losses, fps = False, 0.0, [], [], []
    while not done:
        if is_sb3:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = agent.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        rtts.append(obs[0])
        losses.append(obs[2] * 100.0)
        fps.append(obs[4])
        done = terminated or truncated
    return {"reward": total, "rtt": float(np.mean(rtts)),
            "loss": float(np.mean(losses)), "fps": float(np.mean(fps))}
 
 
def evaluate_policy(agent, is_sb3=False, seeds=EVAL_SEEDS):
    env = NetworkEnv()
    eps = [run_episode(env, agent, s, is_sb3) for s in seeds]
    out = {k: (float(np.mean([e[k] for e in eps])),
               float(1.96 * np.std([e[k] for e in eps], ddof=1) / np.sqrt(len(eps))))
           for k in eps[0]}
    out["episode_rewards"] = np.array([e["reward"] for e in eps])
    return out
 
 
# grid search over constant (fec, delay), scored on the tuning seeds only
def tune_fixed():
    best = None
    for fec in np.arange(0.0, 0.5001, 0.05):
        for delay in np.arange(4.0, 9.0001, 0.25):
            res = evaluate_policy(FixedPacingBaseline(float(fec), float(delay)), seeds=TUNE_SEEDS)
            if best is None or res["reward"][0] > best[0]:
                best = (res["reward"][0], float(fec), float(delay))
    return best
 
 
# result is cached and redone automatically if network_env.py changes
def get_tuned_fixed(force=False):
    env_hash = hashlib.md5(open("network_env.py", "rb").read()).hexdigest()
    if not force and os.path.exists(TUNED_CACHE):
        try:
            cached = json.load(open(TUNED_CACHE))
            if cached.get("env_hash") == env_hash:
                return FixedPacingBaseline(cached["fec"], cached["delay"])
        except (ValueError, KeyError):
            pass
    print("Tuning the fixed baseline by grid search (about a minute, runs once)...")
    reward, fec, delay = tune_fixed()
    os.makedirs("results", exist_ok=True)
    json.dump({"env_hash": env_hash, "fec": fec, "delay": delay, "tune_reward": reward},
              open(TUNED_CACHE, "w"))
    print(f"Tuned fixed baseline: FEC={fec:.2f}, delay={delay:.2f} ms")
    return FixedPacingBaseline(fec, delay)
 
 
def print_result(name, res):
    print(f"[{name}] {len(EVAL_SEEDS)} seeded episodes:")
    print(f"   Mean Episode Reward : {res['reward'][0]:.1f} (95% CI +/- {res['reward'][1]:.1f})")
    print(f"   Avg RTT             : {res['rtt'][0]:.1f} ms")
    print(f"   Avg Effective Loss  : {res['loss'][0]:.2f} %")
    print(f"   Avg Render FPS      : {res['fps'][0]:.1f}\n")
 
 
if __name__ == "__main__":
    print("--- SignalSync: baselines vs. RL agent (same seeded episodes for all) ---\n")
    tuned = get_tuned_fixed()
    print_result(f"Fixed naive (FEC {NAIVE_FEC}, {NAIVE_DELAY} ms)", evaluate_policy(FixedPacingBaseline()))
    print_result(f"Fixed tuned (FEC {tuned.fec_rate:.2f}, {tuned.pacing_delay:.2f} ms)", evaluate_policy(tuned))
    print_result("AIMD heuristic", evaluate_policy(AIMDBaseline()))
    ppo = load_ppo()
    if ppo is not None:
        print_result("PPO SignalSync", evaluate_policy(ppo, is_sb3=True))