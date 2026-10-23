import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from network_env import SignalSyncEnv
from baselines import FixedPacingBaseline, AIMDBaseline

def collect_metrics(agent, is_sb3=False, num_episodes=30, base_seed=100):
    env = SignalSyncEnv()
    ep_rewards, ep_rtts, ep_losses, ep_fps = [], [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        total_rew = 0.0
        rtts, losses, fps_vals = [], [], []

        while not done:
            if is_sb3:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_rew += reward
            rtts.append(obs[0])
            losses.append(obs[2] * 100.0)
            fps_vals.append(obs[4])
            done = terminated or truncated

        ep_rewards.append(total_rew)
        ep_rtts.append(np.mean(rtts))
        ep_losses.append(np.mean(losses))
        ep_fps.append(np.mean(fps_vals))

    return {
        "reward": (np.mean(ep_rewards), np.std(ep_rewards)),
        "rtt": (np.mean(ep_rtts), np.std(ep_rtts)),
        "loss": (np.mean(ep_losses), np.std(ep_losses)),
        "fps": (np.mean(ep_fps), np.std(ep_fps)),
    }

def collect_single_trace(agent, is_sb3=False, seed=42):
    env = SignalSyncEnv()
    obs, _ = env.reset(seed=seed)
    done = False
    rtts, losses, buffers, fps_vals, fecs, delays = [], [], [], [], [], []

    while not done:
        if is_sb3:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = agent.predict(obs)
        fec, delay = SignalSyncEnv.decode_action(action)
        obs, _, terminated, truncated, _ = env.step(action)

        rtts.append(obs[0])
        losses.append(obs[2] * 100.0)
        buffers.append(obs[3])
        fps_vals.append(obs[4])
        fecs.append(fec * 100.0)
        delays.append(delay)
        done = terminated or truncated

    return {
        "rtt": np.array(rtts),
        "loss": np.array(losses),
        "buffer": np.array(buffers),
        "fps": np.array(fps_vals),
        "fec": np.array(fecs),
        "delay": np.array(delays),
    }

def main():
    os.makedirs("results", exist_ok=True)
    model_path = "models/ppo_signalsync.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run `python train.py` first.")

    fixed_agent = FixedPacingBaseline(fec_rate=0.15, pacing_delay=4.5)
    aimd_agent = AIMDBaseline()
    ppo_agent = PPO.load(model_path)

    print("Running 30-episode evaluation across all protocols...")
    agents = [
        ("Fixed (UDP)", fixed_agent, False, "#e74c3c"),
        ("AIMD (TCP)", aimd_agent, False, "#f39c12"),
        ("SignalSync (PPO)", ppo_agent, True, "#2ecc71"),
    ]

    results = {}
    for name, agent, is_sb3, _ in agents:
        results[name] = collect_metrics(agent, is_sb3=is_sb3, num_episodes=30)

    # --- Plot 1: 4-Panel Bar Chart Summary ---
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    fig.suptitle("SignalSync (PPO) vs. Traditional Networking Baselines (30 Episodes)", fontsize=14, fontweight="bold")

    metrics_cfg = [
        ("reward", "Mean Episode Reward (Higher is Better)", axes[0, 0]),
        ("rtt", "Average Latency / RTT in ms (Lower is Better)", axes[0, 1]),
        ("loss", "Effective Packet Loss % (Lower is Better)", axes[1, 0]),
        ("fps", "Client Render FPS (Higher is Better)", axes[1, 1]),
    ]

    names = [a[0] for a in agents]
    colors = [a[3] for a in agents]

    for key, title, ax in metrics_cfg:
        means = [results[n][key][0] for n in names]
        stds = [results[n][key][1] for n in names]
        bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, edgecolor="black", alpha=0.88)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, mean in zip(bars, means):
            y_pos = bar.get_height()
            offset = -0.08 * abs(y_pos) if y_pos < 0 else 0.03 * max(means)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos + offset,
                f"{mean:.1f}",
                ha="center",
                va="bottom" if y_pos >= 0 else "top",
                fontweight="bold",
                fontsize=9,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    bar_path = "results/benchmark_comparison.png"
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved bar chart to {bar_path}")

    # --- Plot 2: Single-Episode Time-Series Trace ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, dpi=300)
    fig.suptitle("Real-Time Protocol Behavior Over a 200-Step Network Trace", fontsize=14, fontweight="bold")

    for name, agent, is_sb3, color in agents:
        # Re-instantiate AIMD so internal delay resets cleanly
        if "AIMD" in name:
            agent = AIMDBaseline()
        trace = collect_single_trace(agent, is_sb3=is_sb3, seed=42)
        steps = np.arange(1, len(trace["rtt"]) + 1)

        axes[0].plot(steps, trace["rtt"], label=name, color=color, linewidth=1.8)
        axes[1].plot(steps, trace["loss"], label=name, color=color, linewidth=1.8)
        axes[2].plot(steps, trace["fps"], label=name, color=color, linewidth=1.8)

    axes[0].set_ylabel("RTT Latency (ms)", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_ylabel("Effective Loss (%)", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].set_ylabel("Render FPS", fontweight="bold")
    axes[2].set_xlabel("Simulation Step (100ms ticks)", fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    trace_path = "results/episode_timeseries.png"
    plt.savefig(trace_path)
    plt.close()
    print(f"Saved time-series trace to {trace_path}")

if __name__ == "__main__":
    main()