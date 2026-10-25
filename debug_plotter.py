import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from stable_baselines3 import PPO

from network_env import SignalSyncEnv
from baselines import FixedPacingBaseline, AIMDBaseline

class RealTimeNetworkVisualizer:
    def __init__(self, steps=300):
        self.steps = steps
        self.env = SignalSyncEnv(max_steps=steps)
        self.is_running = True
        
        # Load Baseline Agents
        self.fixed_agent = FixedPacingBaseline(fec_rate=0.15, pacing_delay=4.5)
        self.aimd_agent = AIMDBaseline()
        
        # Load Trained PPO Agent
        model_path = "models/ppo_signalsync.zip"
        if os.path.exists(model_path):
            try:
                self.ppo_agent = PPO.load(model_path)
                print("Loaded trained PPO model successfully.")
                self.current_agent_name = "SignalSync (PPO)"
            except Exception as e:
                print(f"Failed to load model file ({e}). Defaulting to AIMD.")
                self.ppo_agent = None
                self.current_agent_name = "AIMD (TCP)"
        else:
            self.ppo_agent = None
            print(f"Warning: '{model_path}' not found. Defaulting to AIMD Baseline.")
            self.current_agent_name = "AIMD (TCP)"

        # Data Buffers
        self.time_axis = []
        self.rtts = []
        self.losses = []
        self.fps_list = []

    def get_action(self, obs):
        if self.current_agent_name == "SignalSync (PPO)" and self.ppo_agent:
            action, _ = self.ppo_agent.predict(obs, deterministic=True)
            return action
        elif self.current_agent_name == "AIMD (TCP)":
            return self.aimd_agent.predict(obs)
        else:
            return self.fixed_agent.predict(obs)

    def _on_close(self, event):
        """Cleanly handle figure close events to stop execution loop."""
        self.is_running = False

    def run(self):
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle("SignalSync Real-Time Network Pacing Dashboard", fontsize=14, fontweight="bold")
        
        # Register close callback
        fig.canvas.mpl_connect("close_event", self._on_close)

        # Subplots Layout
        gs = fig.add_gridspec(3, 2, width_ratios=[4, 1])
        ax_rtt = fig.add_subplot(gs[0, 0])
        ax_loss = fig.add_subplot(gs[1, 0], sharex=ax_rtt)
        ax_fps = fig.add_subplot(gs[2, 0], sharex=ax_rtt)
        ax_radio = fig.add_subplot(gs[:, 1])

        # Pre-allocate line objects for efficient zero-flicker redraws
        (line_rtt,) = ax_rtt.plot([], [], color="#2980b9", linewidth=2, label="RTT Latency (ms)")
        (line_loss,) = ax_loss.plot([], [], color="#c0392b", linewidth=2, label="Effective Loss (%)")
        (line_fps,) = ax_fps.plot([], [], color="#27ae60", linewidth=2, label="Client Render FPS")

        ax_rtt.set_ylabel("RTT (ms)", fontweight="bold")
        ax_rtt.grid(True, linestyle="--", alpha=0.5)
        ax_rtt.legend(loc="upper left")

        ax_loss.set_ylabel("Loss (%)", fontweight="bold")
        ax_loss.set_ylim(-2, 102)
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        ax_loss.legend(loc="upper left")

        ax_fps.set_ylabel("FPS", fontweight="bold")
        ax_fps.set_xlabel("Simulation Step (100ms Ticks)", fontweight="bold")
        ax_fps.set_ylim(0, 65)
        ax_fps.grid(True, linestyle="--", alpha=0.5)
        ax_fps.legend(loc="lower left")

        # Interactive Policy Radio Buttons
        ax_radio.set_title("Active Protocol", fontweight="bold", fontsize=11)
        active_idx = 0 if (self.current_agent_name == "SignalSync (PPO)" and self.ppo_agent) else 1
        radio = RadioButtons(
            ax_radio,
            ("SignalSync (PPO)", "AIMD (TCP)", "Fixed (UDP)"),
            active=active_idx,
            activecolor="#2ecc71"
        )

        def switch_agent(label):
            if label == "SignalSync (PPO)" and self.ppo_agent is None:
                print("Cannot switch to PPO: Model file 'models/ppo_signalsync.zip' is missing.")
                return
            self.current_agent_name = label
            if label == "AIMD (TCP)":
                self.aimd_agent = AIMDBaseline()
            print(f"Switched network policy controller to: {label}")

        radio.on_clicked(switch_agent)

        plt.ion()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)

        obs, info = self.env.reset(seed=42)

        for step in range(1, self.steps + 1):
            if not self.is_running or not plt.fignum_exists(fig.number):
                break

            action = self.get_action(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)

            # Map Gymnasium state vector elements accurately
            # State S: [RTT (0), Jitter (1), Effective Loss (2), Buffer Fill (3), Render FPS (4)]
            rtt = float(obs[0])
            loss = float(obs[2] * 100.0)
            fps = float(obs[4])

            self.time_axis.append(step)
            self.rtts.append(rtt)
            self.losses.append(loss)
            self.fps_list.append(fps)

            # Fast direct array updates without rebuilding axes
            line_rtt.set_data(self.time_axis, self.rtts)
            line_loss.set_data(self.time_axis, self.losses)
            line_fps.set_data(self.time_axis, self.fps_list)

            # Auto-scale x-axis and y-axis dynamically
            ax_rtt.set_xlim(1, max(50, step))
            ax_rtt.set_ylim(0, max(80.0, max(self.rtts) + 15.0))

            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.01)
            except Exception:
                break

            if terminated or truncated:
                obs, info = self.env.reset()

        plt.ioff()
        print("Visualization complete.")
        if plt.fignum_exists(fig.number):
            plt.show()

if __name__ == "__main__":
    visualizer = RealTimeNetworkVisualizer(steps=300)
    visualizer.run()