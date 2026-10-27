import numpy as np
from baselines import *

tuned, ppo, env = get_tuned_fixed(), load_ppo(), NetworkEnv()
a = np.array([run_episode(env, tuned, s)["reward"] for s in EVAL_SEEDS])
b = np.array([run_episode(env, ppo, s, True)["reward"] for s in EVAL_SEEDS])
d = b - a
print("mean gain", d.mean(), "+/-", d.std(ddof=1) / np.sqrt(len(d)), "|SignalSync better in", (d > 0).sum(), "of", len(d))