# 📡 SignalSync: RL-Based Adaptive Packet Pacing
**Target:** AI Assignment 2 (Mini Project) | **Status:** Active Development

## 🎯 Project Goal
To build a continuous-action Reinforcement Learning agent (PPO/SAC) that dynamically adjusts Forward Error Correction (FEC) and packet pacing to optimize ultra-low latency multiplayer gaming environments over congested, stochastic networks.

## 📈 Overall Progress
**Phase:** 1 / 6 | **Completion:** 5%

---

## 🛠 Execution Plan & Task Tracker

### Phase 1: Foundation & Logistics (Status: 100%)
- [x] Submit 50-word FCFS abstract.
- [x] Initialize private GitHub repository.
- [x] Define overarching architecture and set up this master execution plan.

### Phase 2: The Network Simulator / Gym Environment (Status: 60%)
*This is the core of the simulation required for the bonus.*
- [x] Define the State (`S`), Action (`A`), and Reward (`R`) mathematically in code.
- [x] Implement the `__init__`, `reset`, and `step` functions in `network_env.py`.
- [x] Build a "Token Bucket" pacing mechanism to simulate network socket buffers.
- [ ] Implement a Markov Chain generator to simulate bursty packet loss and jitter spikes.
- [ ] **Test:** Ensure `env.step()` runs 10,000 random actions without crashing or throwing infinite values.

### Phase 3: RL Agent Implementation (Status: 40%)
- [x] Set up `stable-baselines3` integration in `train.py`.
- [x] Implement Proximal Policy Optimization (PPO) configured for continuous action spaces.
- [ ] Set up TensorBoard logging to track reward curves, agent latency, and packet loss over time.
- [ ] Train the initial agent (baseline model).

### Phase 4: Benchmarking & Baselines (Status: 0%)
*To prove the RL agent is actually smart, we must compare it to standard protocols.*
- [ ] Write a "Naive UDP" baseline (action: always pace 0ms, FEC 0).
- [ ] Write a "Simple AIMD" baseline (Additive Increase, Multiplicative Decrease - similar to TCP Reno).
- [ ] Run evaluation scripts to compare the RL Agent vs. UDP vs. AIMD over 100 episodes.
- [ ] Generate comparative plots (Latency vs. Throughput) and save to `/results/`.

### Phase 5: The Visualizer Demo (Status: 0%)
*Crucial for the resume and presentation.*
- [ ] Write `simulate_gui.py` using Pygame or Matplotlib to visualize the network queue in real-time.
- [ ] Add live metrics to the GUI (Current RTT, Buffer Level, Agent Action).
- [ ] Record a GIF of the RL agent successfully recovering from a network congestion spike.
- [ ] Embed the GIF in this README.

### Phase 6: The Overleaf Report (Status: 0%)
*Strict 2-page limit, two-column format.*
- [ ] Draft Section 1: Abstract & Real-World Problem Mapping.
- [ ] Draft Section 2: Mathematical Formulation (MDP, State, Action, Reward).
- [ ] Draft Section 3: Environment Dynamics (How the network queue is simulated).
- [ ] Draft Section 4: Results & Baseline Comparisons (Include TensorBoard/Matplotlib graphs).
- [ ] Final proofread against the plagiarism and rubric requirements.

### Phase 7: Delivery & Final Polish (Status: 0%)
- [ ] Clean up all Python code (remove debug prints, add docstrings).
- [ ] Finalize `requirements.txt` and add exact run instructions to the README.
- [ ] Ensure commit history shows clear, distributed effort from the team.
