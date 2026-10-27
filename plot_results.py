import glob
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
 
from baselines import (AIMDBaseline, EVAL_SEEDS, FixedPacingBaseline, evaluate_policy,
                       get_tuned_fixed, load_ppo)
from network_env import SignalSyncEnv
 
TRACE_SEED = 42
 
 
# runs one seeded episode and records per-step metrics for the time-series plot
def collect_single_trace(agent, is_sb3=False, seed=TRACE_SEED):
    env = SignalSyncEnv()
    obs, _ = env.reset(seed=seed)
    if not is_sb3:
        agent.reset()
    done = False
    t = {k: [] for k in ("rtt", "loss", "buffer", "fps", "fec", "delay")}
    while not done:
        if is_sb3:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = agent.predict(obs)
        fec, delay = SignalSyncEnv.decode_action(action)
        obs, _, terminated, truncated, _ = env.step(action)
        t["rtt"].append(obs[0]); t["loss"].append(obs[2] * 100.0)
        t["buffer"].append(obs[3]); t["fps"].append(obs[4])
        t["fec"].append(fec * 100.0); t["delay"].append(delay)
        done = terminated or truncated
    return {k: np.array(v) for k, v in t.items()}
 
 
# training reward per seed, from the Monitor logs written by train.py
def plot_learning_curves():
    files = sorted(glob.glob("results/train_monitor_seed*.monitor.csv"))
    if not files:
        print("No training logs found - skipping learning curve (run train.py).")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    for f in files:
        data = np.atleast_2d(np.genfromtxt(f, delimiter=",", skip_header=2))
        rewards, steps = data[:, 0], np.cumsum(data[:, 1])
        w = max(1, min(10, len(rewards)))
        smooth = np.convolve(rewards, np.ones(w) / w, mode="valid")
        label = os.path.basename(f).split(".")[0].replace("train_monitor_", "")
        ax.plot(steps[w - 1:], smooth, label=label)
    ax.set_xlabel("Training timesteps"); ax.set_ylabel("Episode reward (10-episode average)")
    ax.set_title("PPO learning curves"); ax.grid(True, alpha=0.4); ax.legend()
    plt.tight_layout(); plt.savefig("results/learning_curve.png"); plt.close()
    print("Saved results/learning_curve.png")
 
 
def main():
    os.makedirs("results", exist_ok=True)
    tuned = get_tuned_fixed()  # best constant policy found by grid search
    ppo = load_ppo()
    agents = [
        ("Fixed (naive)", FixedPacingBaseline(), False, "#e74c3c"),
        (f"Fixed (tuned {tuned.fec_rate:.2f}/{tuned.pacing_delay:.2f}ms)", tuned, False, "#3498db"),
        ("AIMD", AIMDBaseline(), False, "#f39c12"),
    ]
    if ppo is not None:
        agents.append(("SignalSync (PPO)", ppo, True, "#2ecc71"))
    else:
        print("\n*** PPO missing: figures below contain baselines ONLY. ***\n")
 
    print(f"Evaluating {len(agents)} policies on the same {len(EVAL_SEEDS)} seeded episodes...")
    # every policy runs on the same seeded episodes, so the networks are identical
    results = {n: evaluate_policy(a, is_sb3=s) for n, a, s, _ in agents}
 
    # results table for the report
    lines = [f"{'Policy':44s} {'Reward':>16s} {'RTT ms':>9s} {'Loss %':>8s} {'FPS':>7s}"]
    for n, r in results.items():
        lines.append(f"{n:44s} {r['reward'][0]:8.1f} +/- {r['reward'][1]:5.1f} "
                     f"{r['rtt'][0]:9.1f} {r['loss'][0]:8.2f} {r['fps'][0]:7.1f}")
    table = "\n".join(lines)
    print("\n" + table)
    open("results/results_table.txt", "w").write(table + "\n")
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=300)
    fig.suptitle(f"SignalSync vs. baselines ({len(EVAL_SEEDS)} seeded episodes, same networks)",
                 fontsize=14, fontweight="bold")
    cfg = [("reward", "Mean Episode Reward (higher is better)", axes[0, 0]),
           ("rtt", "Average RTT in ms (lower is better)", axes[0, 1]),
           ("loss", "Effective Packet Loss % (lower is better)", axes[1, 0]),
           ("fps", "Client Render FPS (higher is better)", axes[1, 1])]
    names = [a[0] for a in agents]; colors = [a[3] for a in agents]
    for key, title, ax in cfg:
        means = [results[n][key][0] for n in names]
        stds = [results[n][key][1] for n in names]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=6, color=colors,
                      edgecolor="black", alpha=0.88)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if min(means) < 0:
            low = min(m - s for m, s in zip(means, stds)); ax.set_ylim(low * 1.18, 0); span = abs(low)
        else:
            high = max(m + s for m, s in zip(means, stds)); ax.set_ylim(0, high * 1.18); span = high
        for bar, mean, std in zip(bars, means, stds):
            y = (mean - std) - 0.03 * span if mean < 0 else (mean + std) + 0.02 * span
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{mean:.1f}", ha="center",
                    va="top" if mean < 0 else "bottom", fontweight="bold", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig("results/benchmark_comparison.png"); plt.close()
    print("Saved results/benchmark_comparison.png")
 
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=300)
    fig.suptitle(f"Behaviour over one 200-step network trace (seed {TRACE_SEED})",
                 fontsize=14, fontweight="bold")
    for name, agent, is_sb3, color in agents:
        tr = collect_single_trace(agent, is_sb3)
        x = np.arange(1, len(tr["rtt"]) + 1)
        axes[0].plot(x, tr["rtt"], label=name, color=color, linewidth=1.6)
        axes[1].plot(x, tr["loss"], color=color, linewidth=1.6)
        axes[2].plot(x, tr["fps"], color=color, linewidth=1.6)
        axes[3].plot(x, tr["fec"], color=color, linewidth=1.6)
    for ax, lab in zip(axes, ["RTT (ms)", "Effective loss (%)", "Render FPS", "FEC chosen (%)"]):
        ax.set_ylabel(lab, fontweight="bold"); ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    axes[3].set_xlabel("Simulation step (100 ms ticks)", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig("results/episode_timeseries.png"); plt.close()
    print("Saved results/episode_timeseries.png")
 
    plot_learning_curves()
 
 
if __name__ == "__main__":
    main()