import os
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from baselines import FixedPacingBaseline, get_tuned_fixed, load_ppo
from network_env import SignalSyncEnv

SEED = 77
WINDOW = 60
FRAMES = 120
COLORS = {"ppo": "#2ecc71", "tuned": "#3498db", "naive": "#e74c3c"}


def run_visualizer(save_gif=False):
    ppo_model = load_ppo()
    if ppo_model is None:
        raise FileNotFoundError("Model not found. Run `python train.py` first.")

    agents = {"tuned": get_tuned_fixed(), "naive": FixedPacingBaseline()}
    envs = {name: SignalSyncEnv(max_steps=150) for name in COLORS}
    obs = {name: env.reset(seed=SEED)[0] for name, env in envs.items()}
    for agent in agents.values():
        agent.reset()

    steps = []
    hist = {name: {"rtt": [], "loss": [], "fps": []} for name in COLORS}

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 7), dpi=120)
    fig.suptitle("SignalSync: Live RL Adaptive Packet Pacing Telemetry",
                 fontsize=15, fontweight="bold", color="#00ffcc")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 2.2], hspace=0.45, wspace=0.35)
    ax_buf = fig.add_subplot(gs[0:2, 0])
    ax_act = fig.add_subplot(gs[2, 0])
    ax_rtt = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, 1])
    ax_fps = fig.add_subplot(gs[2, 1])

    def update(frame):
        step = len(steps) + 1
        steps.append(step)

        action_ppo, _ = ppo_model.predict(obs["ppo"], deterministic=True)
        fec_ppo, delay_ppo = SignalSyncEnv.decode_action(action_ppo)
        obs["ppo"] = envs["ppo"].step(action_ppo)[0]
        for name, agent in agents.items():
            obs[name] = envs[name].step(agent.predict(obs[name]))[0]

        for name in COLORS:
            hist[name]["rtt"].append(obs[name][0])
            hist[name]["loss"].append(obs[name][2] * 100.0)
            hist[name]["fps"].append(obs[name][4])

        x = steps[-WINDOW:]
        bad = envs["ppo"].channel.current_state == 1

        ax_buf.clear()
        names = ["PPO", "Tuned", "Naive"]
        vals = [obs["ppo"][3], obs["tuned"][3], obs["naive"][3]]
        bars = ax_buf.bar(names, vals, color=[COLORS["ppo"], COLORS["tuned"], COLORS["naive"]], width=0.5)
        ax_buf.set_ylim(0, 105)
        ax_buf.set_ylabel("Packets in Queue")
        ax_buf.set_title("Link: " + ("CONGESTION BURST (BAD)" if bad else "HEALTHY LINK (GOOD)"),
                         color="#ff4d4d" if bad else "#2ecc71", fontweight="bold", fontsize=10)
        for b, v in zip(bars, vals):
            ax_buf.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontweight="bold")

        ax_act.clear()
        act_vals = [fec_ppo * 100.0, delay_ppo]
        ax_act.barh(["FEC (%)", "Delay (ms)"], act_vals, color=["#3498db", "#9b59b6"])
        ax_act.set_xlim(0, 55)
        ax_act.set_title("PPO Agent Live Actions", fontsize=10, fontweight="bold")
        for i, v in enumerate(act_vals):
            ax_act.text(v + 1.5, i, f"{v:.1f}", va="center", fontweight="bold")

        labels = {"ppo": "SignalSync PPO", "tuned": "Fixed (tuned)", "naive": "Fixed (naive)"}
        panels = [(ax_rtt, "rtt", "RTT (ms)", (0, 240)),
                  (ax_loss, "loss", "Loss (%)", (-2, 50)),
                  (ax_fps, "fps", "Client FPS", (0, 65))]
        for ax, key, ylabel, ylim in panels:
            ax.clear()
            for name in ("naive", "tuned", "ppo"):
                ax.plot(x, hist[name][key][-WINDOW:], color=COLORS[name], label=labels[name],
                        linewidth=2 if name == "ppo" else 1.3,
                        linestyle="--" if name == "naive" else "-")
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.2)
        ax_rtt.legend(loc="upper left", fontsize=8)
        ax_fps.set_xlabel("Simulation Step (100ms ticks)")

    ani = FuncAnimation(fig, update, frames=FRAMES, interval=80, repeat=False, init_func=lambda: None)

    if save_gif:
        os.makedirs("results", exist_ok=True)
        gif_path = "results/signalsync_demo.gif"
        print(f"Rendering {FRAMES}-frame GIF to {gif_path} (takes a minute)...")
        ani.save(gif_path, writer=PillowWriter(fps=12))
        print(f"Saved GIF to {gif_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    run_visualizer(save_gif="--save-gif" in sys.argv)