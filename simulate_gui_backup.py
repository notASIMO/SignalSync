import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
from network_env import SignalSyncEnv
from baselines import FixedPacingBaseline

def run_visualizer(save_gif=False):
    model_path = "models/ppo_signalsync.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run `python train.py` first.")

    ppo_model = PPO.load(model_path)
    udp_agent = FixedPacingBaseline(fec_rate=0.10, pacing_delay=4.2)

    env_ppo = SignalSyncEnv(max_steps=150)
    env_udp = SignalSyncEnv(max_steps=150)

    obs_ppo, _ = env_ppo.reset(seed=77)
    obs_udp, _ = env_udp.reset(seed=77)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 7), dpi=120)
    fig.suptitle("SignalSync: Live RL Adaptive Packet Pacing Telemetry", fontsize=15, fontweight="bold", color="#00ffcc")

    gs = fig.add_gridspec(3, 3, width_ratios=[1.1, 2.2, 0.05], hspace=0.45, wspace=0.35)

    ax_buf = fig.add_subplot(gs[0:2, 0])
    ax_act = fig.add_subplot(gs[2, 0])
    ax_rtt = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, 1])
    ax_fps = fig.add_subplot(gs[2, 1])

    steps_hist = []
    ppo_rtt_hist, udp_rtt_hist = [], []
    ppo_loss_hist, udp_loss_hist = [], []
    ppo_fps_hist, udp_fps_hist = [], []

    state_holder = {"obs_ppo": obs_ppo, "obs_udp": obs_udp, "step": 0}

    def update(frame):
        s = state_holder["step"] + 1
        state_holder["step"] = s

        # Step PPO and UDP with synchronized Markov channel transitions
        np_state = np.random.get_state()
        action_ppo, _ = ppo_model.predict(state_holder["obs_ppo"], deterministic=True)
        fec_ppo, delay_ppo = SignalSyncEnv.decode_action(action_ppo)
        obs_p, _, _, _, _ = env_ppo.step(action_ppo)

        np.random.set_state(np_state)
        action_udp = udp_agent.predict(state_holder["obs_udp"])
        obs_u, _, _, _, _ = env_udp.step(action_udp)

        state_holder["obs_ppo"] = obs_p
        state_holder["obs_udp"] = obs_u

        steps_hist.append(s)
        ppo_rtt_hist.append(obs_p[0])
        udp_rtt_hist.append(obs_u[0])
        ppo_loss_hist.append(obs_p[2] * 100.0)
        udp_loss_hist.append(obs_u[2] * 100.0)
        ppo_fps_hist.append(obs_p[4])
        udp_fps_hist.append(obs_u[4])

        # Keep rolling 60-step window
        win = 60
        x_win = steps_hist[-win:]

        # 1. Token Bucket Buffer Bar (PPO vs UDP)
        ax_buf.clear()
        buf_vals = [obs_p[3], obs_u[3]]
        buf_colors = ["#2ecc71" if obs_p[3] < 35 else "#f1c40f", "#e74c3c"]
        bars = ax_buf.bar(["PPO Buffer", "UDP Buffer"], buf_vals, color=buf_colors, width=0.5)
        ax_buf.set_ylim(0, 105)
        ax_buf.set_ylabel("Packets in Queue")
        markov_str = "CONGESTION BURST (BAD)" if env_ppo.channel.current_state == 1 else "HEALTHY LINK (GOOD)"
        markov_col = "#ff4d4d" if env_ppo.channel.current_state == 1 else "#2ecc71"
        ax_buf.set_title(f"Link: {markov_str}", color=markov_col, fontweight="bold", fontsize=10)
        for b, v in zip(bars, buf_vals):
            ax_buf.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f} pkts", ha="center", fontweight="bold")

        # 2. PPO Live Action Gauges
        ax_act.clear()
        act_vals = [fec_ppo * 100.0, delay_ppo]
        ax_act.barh(["FEC (%)", "Delay (ms)"], act_vals, color=["#3498db", "#9b59b6"])
        ax_act.set_xlim(0, 55)
        ax_act.set_title("PPO Agent Live Actions", fontsize=10, fontweight="bold")
        for i, v in enumerate(act_vals):
            ax_act.text(v + 1.5, i, f"{v:.1f}", va="center", fontweight="bold")

        # 3. Scrolling Telemetry Traces
        ax_rtt.clear()
        ax_rtt.plot(x_win, udp_rtt_hist[-win:], color="#e74c3c", linestyle="--", label="Naive UDP")
        ax_rtt.plot(x_win, ppo_rtt_hist[-win:], color="#2ecc71", linewidth=2, label="SignalSync PPO")
        ax_rtt.set_ylabel("RTT (ms)")
        ax_rtt.set_ylim(0, 240)
        ax_rtt.legend(loc="upper left", fontsize=8)
        ax_rtt.grid(True, alpha=0.2)

        ax_loss.clear()
        ax_loss.plot(x_win, udp_loss_hist[-win:], color="#e74c3c", linestyle="--")
        ax_loss.plot(x_win, ppo_loss_hist[-win:], color="#2ecc71", linewidth=2)
        ax_loss.set_ylabel("Loss (%)")
        ax_loss.set_ylim(-2, 50)
        ax_loss.grid(True, alpha=0.2)

        ax_fps.clear()
        ax_fps.plot(x_win, udp_fps_hist[-win:], color="#e74c3c", linestyle="--")
        ax_fps.plot(x_win, ppo_fps_hist[-win:], color="#2ecc71", linewidth=2)
        ax_fps.set_ylabel("Client FPS")
        ax_fps.set_xlabel("Simulation Step (100ms ticks)")
        ax_fps.set_ylim(0, 65)
        ax_fps.grid(True, alpha=0.2)

    ani = FuncAnimation(fig, update, frames=120, interval=80, repeat=False)

    if save_gif:
        os.makedirs("results", exist_ok=True)
        gif_path = "results/signalsync_demo.gif"
        print(f"Rendering 120-frame live telemetry GIF to {gif_path} (takes ~15 seconds)...")
        ani.save(gif_path, writer=PillowWriter(fps=12))
        print(f"Saved GIF successfully to {gif_path}!")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    save_mode = "--save-gif" in sys.argv
    run_visualizer(save_gif=save_mode)