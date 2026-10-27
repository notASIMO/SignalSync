import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from network_env import SignalSyncEnv
from baselines import AIMDBaseline, FixedPacingBaseline, get_tuned_fixed, load_ppo

# same seed -> same random network after every reset, change it to see a different one
SEED = 42


class SimulationManager:
    def __init__(self):
        self.envs = {n: SignalSyncEnv(max_steps=100000) for n in ("ppo", "tuned", "udp", "aimd")}
        self.ppo_model = load_ppo()
        self.agents = {
            "tuned": get_tuned_fixed(),
            "udp": FixedPacingBaseline(),
            "aimd": AIMDBaseline(),
        }
        self.reset()

    def reset(self):
        self.obs = {n: env.reset(seed=SEED)[0] for n, env in self.envs.items()}
        for agent in self.agents.values():
            agent.reset()
        self.step_count = 0

    def trigger_congestion(self):
        for env in self.envs.values():
            env.channel.current_state = 1

    def step(self, manual_mode=False, manual_fec=0.2, manual_delay=6.0):
        self.step_count += 1

        if manual_mode or self.ppo_model is None:
            action_ppo = SignalSyncEnv.encode_action(manual_fec, manual_delay)
        else:
            action_ppo, _ = self.ppo_model.predict(self.obs["ppo"], deterministic=True)

        fec_ppo, delay_ppo = SignalSyncEnv.decode_action(action_ppo)
        self.obs["ppo"], rew_ppo, _, _, _ = self.envs["ppo"].step(action_ppo)
        markov_state = self.envs["ppo"].channel.current_state

        for name, agent in self.agents.items():
            self.obs[name], _, _, _, _ = self.envs[name].step(agent.predict(self.obs[name]))

        obs_ppo = self.obs["ppo"]
        if manual_mode:
            orb_state = "listening"
            orb_label = "MANUAL OVERRIDE ACTIVE — Monitoring user pacing parameters"
        elif markov_state == 1:
            orb_state = "solving"
            orb_label = "SOLVING CONGESTION BURST — Boosting FEC shield & pacing delay"
        elif obs_ppo[3] > 15.0:
            orb_state = "searching"
            orb_label = "DRAINING SOCKET BUFFER — Scanning optimal queue equilibrium"
        else:
            orb_state = "working"
            orb_label = "OPTIMAL PACING LOCKED — Ultra-low latency 60 FPS stream"

        def snap(obs):
            return {
                "rtt": round(float(obs[0]), 1),
                "loss": round(float(obs[2] * 100.0), 1),
                "buffer": round(float(obs[3]), 1),
                "fps": round(float(obs[4]), 1),
            }

        return {
            "step": self.step_count,
            "markov_bad": int(markov_state),
            "orb_state": orb_state,
            "orb_label": orb_label,
            "ppo": {
                **snap(obs_ppo),
                "jitter": round(float(obs_ppo[1]), 1),
                "fec": round(float(fec_ppo * 100.0), 1),
                "delay": round(float(delay_ppo), 2),
                "reward": round(float(rew_ppo), 2),
            },
            "tuned": snap(self.obs["tuned"]),
            "udp": snap(self.obs["udp"]),
            "aimd": snap(self.obs["aimd"]),
        }


SIM = SimulationManager()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<title>SignalSync — RL Telemetry</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
  body {
    font-family: 'Inter', sans-serif;
    background-color: #06070a;
    color: #f4f4f6;
    overflow-x: hidden;
  }
  .mono { font-family: 'JetBrains Mono', monospace; }
  .kinetic-bg {
    background-size: 36px 36px;
    background-image: 
      linear-gradient(to right, rgba(255,255,255,0.03) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(255,255,255,0.03) 1px, transparent 1px);
  }
  .bento-card {
    background: rgba(15, 17, 23, 0.78);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
  }
  .bento-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    box-shadow: 0 0 28px rgba(56, 189, 248, 0.08);
  }
  .orb-glow-working { filter: drop-shadow(0 0 14px rgba(16, 185, 129, 0.55)); }
  .orb-glow-solving { filter: drop-shadow(0 0 16px rgba(244, 63, 94, 0.7)); }
  .orb-glow-searching { filter: drop-shadow(0 0 14px rgba(245, 158, 11, 0.6)); }
  .orb-glow-listening { filter: drop-shadow(0 0 14px rgba(168, 85, 247, 0.65)); }
</style>
</head>
<body class="kinetic-bg min-h-screen p-5">

  <header class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4 mb-5 bento-card px-6 py-4">
    <div class="flex items-center gap-4">
      <div class="relative flex items-center justify-center w-16 h-16 rounded-2xl bg-black/60 border border-white/10">
        <canvas id="thinkingOrbCanvas" width="64" height="64" class="orb-glow-working"></canvas>
      </div>
      <div>
        <div class="flex items-center gap-2.5">
          <h1 class="text-xl font-bold tracking-tight text-white">SignalSync RL</h1>
          <span id="orbModeBadge" class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 mono">ORB: WORKING</span>
          <span id="linkStateBadge" class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 mono">LINK: GOOD</span>
        </div>
        <p id="orbStatusText" class="text-xs text-slate-400 mt-1 mono">OPTIMAL PACING LOCKED — Ultra-low latency 60 FPS stream</p>
      </div>
    </div>

    <div class="flex flex-wrap items-center gap-3">
      <button id="btnPlayPause" onclick="togglePlay()" class="px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-500 hover:bg-emerald-400 text-black transition">
        ⏸ Pause
      </button>
      <button onclick="injectBurst()" class="px-4 py-2 rounded-xl text-xs font-semibold bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 transition">
        ⚡ Inject Markov Burst
      </button>
      <button onclick="resetSim()" class="px-3.5 py-2 rounded-xl text-xs font-semibold bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 transition">
        ↺ Reset
      </button>
    </div>
  </header>

  <main class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-5">

    <div class="lg:col-span-4 flex flex-col gap-5">

      <div class="bento-card p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Packet Stream</span>
          <span id="stepCounter" class="text-xs mono text-sky-400">Step #0</span>
        </div>
        <canvas id="particleTunnel" width="380" height="150" class="w-full rounded-xl bg-black/70 border border-white/5"></canvas>
        <div class="flex justify-between text-[11px] text-slate-400 mt-2 mono">
          <span>🔵 Base Frame</span>
          <span>🟢 FEC Shield</span>
          <span>🔴 Markov Drop</span>
        </div>
      </div>

      <div class="bento-card p-4">
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Socket Buffer Occupancy (Max 100 pkts)</span>
        <div class="mt-3 space-y-3">
          <div>
            <div class="flex justify-between text-xs mb-1 mono">
              <span class="text-emerald-400 font-semibold">SignalSync PPO (RL)</span>
              <span id="bufPpoText">0.0 pkts</span>
            </div>
            <div class="w-full h-3 bg-white/5 rounded-full overflow-hidden">
              <div id="bufPpoBar" class="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-150" style="width: 0%"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-xs mb-1 mono">
              <span class="text-sky-400 font-semibold">Fixed (tuned)</span>
              <span id="bufTunedText">0.0 pkts</span>
            </div>
            <div class="w-full h-3 bg-white/5 rounded-full overflow-hidden">
              <div id="bufTunedBar" class="h-full bg-sky-500 transition-all duration-150" style="width: 0%"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-xs mb-1 mono">
              <span class="text-amber-400 font-semibold">AIMD Control (TCP)</span>
              <span id="bufAimdText">0.0 pkts</span>
            </div>
            <div class="w-full h-3 bg-white/5 rounded-full overflow-hidden">
              <div id="bufAimdBar" class="h-full bg-amber-500 transition-all duration-150" style="width: 0%"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-xs mb-1 mono">
              <span class="text-rose-400 font-semibold">Fixed (naive)</span>
              <span id="bufUdpText">0.0 pkts</span>
            </div>
            <div class="w-full h-3 bg-white/5 rounded-full overflow-hidden">
              <div id="bufUdpBar" class="h-full bg-rose-500 transition-all duration-150" style="width: 0%"></div>
            </div>
          </div>
        </div>
      </div>

      <div class="bento-card p-4">
        <div class="flex items-center justify-between mb-3">
          <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Agent Continuous Actions</span>
          <label class="flex items-center gap-2 text-xs cursor-pointer text-purple-300">
            <input type="checkbox" id="manualToggle" onchange="toggleManual()" class="accent-purple-500">
            Manual Override
          </label>
        </div>

        <div class="space-y-3">
          <div>
            <div class="flex justify-between text-xs mono mb-1">
              <span>FEC Redundancy Ratio</span>
              <span id="fecValText" class="text-sky-400 font-bold">15.0%</span>
            </div>
            <input type="range" id="fecSlider" min="0" max="50" step="1" value="15" disabled class="w-full accent-sky-400">
          </div>

          <div>
            <div class="flex justify-between text-xs mono mb-1">
              <span>Pacing Delay Interval</span>
              <span id="delayValText" class="text-purple-400 font-bold">6.50 ms</span>
            </div>
            <input type="range" id="delaySlider" min="3.5" max="14.0" step="0.1" value="6.5" disabled class="w-full accent-purple-400">
          </div>
        </div>
      </div>
    </div>

    <div class="lg:col-span-8 flex flex-col gap-5">

      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div class="bento-card p-3.5">
          <div class="text-[11px] text-slate-400 uppercase">PPO Latency (RTT)</div>
          <div id="kpiRtt" class="text-2xl font-bold text-emerald-400 mono mt-1">20.0 ms</div>
          <div id="kpiRttSub" class="text-[11px] text-slate-500 mono mt-0.5">Tuned: 20.0 ms</div>
        </div>
        <div class="bento-card p-3.5">
          <div class="text-[11px] text-slate-400 uppercase">Effective Loss</div>
          <div id="kpiLoss" class="text-2xl font-bold text-sky-400 mono mt-1">0.0 %</div>
          <div id="kpiLossSub" class="text-[11px] text-slate-500 mono mt-0.5">Tuned: 0.0 %</div>
        </div>
        <div class="bento-card p-3.5">
          <div class="text-[11px] text-slate-400 uppercase">Client Render FPS</div>
          <div id="kpiFps" class="text-2xl font-bold text-indigo-400 mono mt-1">60.0 FPS</div>
          <div id="kpiFpsSub" class="text-[11px] text-slate-500 mono mt-0.5">Tuned: 60.0 FPS</div>
        </div>
        <div class="bento-card p-3.5">
          <div class="text-[11px] text-slate-400 uppercase">Step Reward</div>
          <div id="kpiReward" class="text-2xl font-bold text-amber-400 mono mt-1">-0.52</div>
          <div class="text-[11px] text-slate-500 mono mt-0.5">Target: > -1.5</div>
        </div>
      </div>

      <div class="bento-card p-4 flex-1 flex flex-col justify-between gap-4">
        <div class="h-44"><canvas id="rttChart"></canvas></div>
        <div class="h-44"><canvas id="fpsChart"></canvas></div>
      </div>

    </div>
  </main>

<script>
const orbCanvas = document.getElementById('thinkingOrbCanvas');
const orbCtx = orbCanvas.getContext('2d');
let currentOrbState = 'working';
const orbDots = [];
for (let i = 0; i < 42; i++) {
  const phi = Math.acos(1 - 2 * (i + 0.5) / 42);
  const theta = Math.PI * (1 + Math.sqrt(5)) * i;
  orbDots.push({ phi, theta });
}

function drawThinkingOrb(time) {
  const t = time * 0.0022;
  orbCtx.clearRect(0, 0, 64, 64);
  const cx = 32, cy = 32, radius = 22;

  orbDots.forEach((d, idx) => {
    let phi = d.phi;
    let theta = d.theta + t;

    if (currentOrbState === 'solving') {
      theta += Math.sin(t * 3.5 + idx) * 0.65;
      phi += Math.cos(t * 2.5 + idx * 0.5) * 0.25;
    } else if (currentOrbState === 'searching') {
      theta += Math.sin(phi * 3 + t * 2) * 0.3;
    } else if (currentOrbState === 'listening') {
      phi += Math.sin(t * 4 + idx * 0.3) * 0.18;
    }

    const x3d = Math.sin(phi) * Math.cos(theta);
    const y3d = Math.cos(phi);
    const z3d = Math.sin(phi) * Math.sin(theta);

    const tilt = 0.35;
    const yRot = y3d * Math.cos(tilt) - z3d * Math.sin(tilt);
    const zRot = y3d * Math.sin(tilt) + z3d * Math.cos(tilt);

    const scale = (zRot + 2.2) / 3.2;
    const px = cx + x3d * radius * scale;
    const py = cy + yRot * radius * scale;
    const dotRadius = Math.max(0.9, 2.3 * scale);
    const alpha = Math.min(1, Math.max(0.18, (zRot + 1.2) / 2.2));

    let rgb = '52, 211, 153';
    if (currentOrbState === 'solving') rgb = '251, 113, 133';
    else if (currentOrbState === 'searching') rgb = '251, 191, 36';
    else if (currentOrbState === 'listening') rgb = '192, 132, 252';

    orbCtx.beginPath();
    orbCtx.arc(px, py, dotRadius, 0, Math.PI * 2);
    orbCtx.fillStyle = `rgba(${rgb}, ${alpha})`;
    orbCtx.fill();
  });
  requestAnimationFrame(drawThinkingOrb);
}
requestAnimationFrame(drawThinkingOrb);

const tunnelCanvas = document.getElementById('particleTunnel');
const tCtx = tunnelCanvas.getContext('2d');
let packets = [];
let latestState = { ppo: { fec: 15, delay: 6.5, loss: 0, buffer: 0 }, markov_bad: 0 };

function spawnPackets() {
  const count = Math.max(1, Math.round(14 / Math.max(3.5, latestState.ppo.delay)));
  for (let i = 0; i < count; i++) {
    const isFec = Math.random() < (latestState.ppo.fec / 100);
    const isDropped = latestState.markov_bad === 1 && Math.random() < 0.35;
    packets.push({
      x: 15,
      y: 35 + Math.random() * 80,
      vx: 2.6 + Math.random() * 1.4,
      isFec,
      isDropped
    });
  }
}

function drawParticleTunnel() {
  tCtx.fillStyle = 'rgba(8, 10, 15, 0.32)';
  tCtx.fillRect(0, 0, tunnelCanvas.width, tunnelCanvas.height);

  const gateX = 190;
  tCtx.strokeStyle = latestState.markov_bad ? 'rgba(244,63,94,0.6)' : 'rgba(56,189,248,0.3)';
  tCtx.lineWidth = 2;
  tCtx.beginPath();
  tCtx.moveTo(gateX, 15);
  tCtx.lineTo(gateX, 135);
  tCtx.stroke();

  for (let i = packets.length - 1; i >= 0; i--) {
    const p = packets[i];
    p.x += p.vx;

    if (p.isDropped && p.x > gateX) {
      p.y += 2.5;
      tCtx.fillStyle = '#f43f5e';
    } else if (p.isFec) {
      tCtx.fillStyle = '#34d399';
    } else {
      tCtx.fillStyle = '#38bdf8';
    }

    tCtx.beginPath();
    tCtx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
    tCtx.fill();

    if (p.x > tunnelCanvas.width || p.y > tunnelCanvas.height) {
      packets.splice(i, 1);
    }
  }
  requestAnimationFrame(drawParticleTunnel);
}
requestAnimationFrame(drawParticleTunnel);

function createChart(canvasId, label, yMax) {
  return new Chart(document.getElementById(canvasId), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'SignalSync PPO', data: [], borderColor: '#34d399', borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { label: 'Fixed (tuned)', data: [], borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        { label: 'AIMD (TCP)', data: [], borderColor: '#fbbf24', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        { label: 'Fixed (naive)', data: [], borderColor: '#f43f5e', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0, tension: 0.3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 10 } } },
        title: { display: true, text: label, color: '#e2e8f0', font: { size: 11 } }
      },
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { min: 0, max: yMax, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

const rttChart = createChart('rttChart', 'Real-Time RTT Latency (ms) — Lower is Better', 220);
const fpsChart = createChart('fpsChart', 'Client Render Frame Rate (FPS) — Higher is Better', 65);

let isPlaying = true;
function togglePlay() {
  isPlaying = !isPlaying;
  document.getElementById('btnPlayPause').innerText = isPlaying ? '⏸ Pause' : '▶ Resume';
}

function toggleManual() {
  const manual = document.getElementById('manualToggle').checked;
  document.getElementById('fecSlider').disabled = !manual;
  document.getElementById('delaySlider').disabled = !manual;
}

async function injectBurst() {
  await fetch('/api/burst');
}

async function resetSim() {
  await fetch('/api/reset');
  rttChart.data.labels = [];
  fpsChart.data.labels = [];
  rttChart.data.datasets.forEach(d => d.data = []);
  fpsChart.data.datasets.forEach(d => d.data = []);
  rttChart.update();
  fpsChart.update();
}

async function pollStep() {
  if (!isPlaying) return;
  const manual = document.getElementById('manualToggle').checked ? 1 : 0;
  const fec = document.getElementById('fecSlider').value / 100.0;
  const delay = document.getElementById('delaySlider').value;

  const res = await fetch(`/api/step?manual=${manual}&fec=${fec}&delay=${delay}`);
  const data = await res.json();
  latestState = data;
  spawnPackets();

  currentOrbState = data.orb_state;
  orbCanvas.className = `orb-glow-${data.orb_state}`;
  document.getElementById('orbModeBadge').innerText = `ORB: ${data.orb_state.toUpperCase()}`;
  document.getElementById('orbStatusText').innerText = data.orb_label;

  const linkBadge = document.getElementById('linkStateBadge');
  if (data.markov_bad) {
    linkBadge.innerText = 'LINK: BURST LOSS (BAD)';
    linkBadge.className = 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/40 mono';
  } else {
    linkBadge.innerText = 'LINK: HEALTHY (GOOD)';
    linkBadge.className = 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 mono';
  }

  document.getElementById('stepCounter').innerText = `Step #${data.step}`;

  document.getElementById('bufPpoText').innerText = `${data.ppo.buffer} pkts`;
  document.getElementById('bufPpoBar').style.width = `${Math.min(100, data.ppo.buffer)}%`;
  document.getElementById('bufTunedText').innerText = `${data.tuned.buffer} pkts`;
  document.getElementById('bufTunedBar').style.width = `${Math.min(100, data.tuned.buffer)}%`;
  document.getElementById('bufAimdText').innerText = `${data.aimd.buffer} pkts`;
  document.getElementById('bufAimdBar').style.width = `${Math.min(100, data.aimd.buffer)}%`;
  document.getElementById('bufUdpText').innerText = `${data.udp.buffer} pkts`;
  document.getElementById('bufUdpBar').style.width = `${Math.min(100, data.udp.buffer)}%`;

  if (!manual) {
    document.getElementById('fecSlider').value = data.ppo.fec;
    document.getElementById('delaySlider').value = data.ppo.delay;
  }
  document.getElementById('fecValText').innerText = `${data.ppo.fec}%`;
  document.getElementById('delayValText').innerText = `${data.ppo.delay} ms`;

  document.getElementById('kpiRtt').innerText = `${data.ppo.rtt} ms`;
  document.getElementById('kpiRttSub').innerText = `Tuned: ${data.tuned.rtt} ms`;
  document.getElementById('kpiLoss').innerText = `${data.ppo.loss} %`;
  document.getElementById('kpiLossSub').innerText = `Tuned: ${data.tuned.loss} %`;
  document.getElementById('kpiFps').innerText = `${data.ppo.fps} FPS`;
  document.getElementById('kpiFpsSub').innerText = `Tuned: ${data.tuned.fps} FPS`;
  document.getElementById('kpiReward').innerText = `${data.ppo.reward}`;

  const pushData = (chart, key) => {
    chart.data.labels.push(data.step);
    chart.data.datasets[0].data.push(data.ppo[key]);
    chart.data.datasets[1].data.push(data.tuned[key]);
    chart.data.datasets[2].data.push(data.aimd[key]);
    chart.data.datasets[3].data.push(data.udp[key]);
    if (chart.data.labels.length > 50) {
      chart.data.labels.shift();
      chart.data.datasets.forEach(d => d.data.shift());
    }
    chart.update();
  };
  pushData(rttChart, 'rtt');
  pushData(fpsChart, 'fps');
}

setInterval(pollStep, 140);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/step":
            qs = parse_qs(parsed.query)
            manual = qs.get("manual", ["0"])[0] == "1"
            fec = float(qs.get("fec", ["0.2"])[0])
            delay = float(qs.get("delay", ["6.0"])[0])
            payload = SIM.step(manual_mode=manual, manual_fec=fec, manual_delay=delay)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif parsed.path == "/api/burst":
            SIM.trigger_congestion()
            self.send_response(200)
            self.end_headers()
        elif parsed.path == "/api/reset":
            SIM.reset()
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    port = 8050
    url = f"http://localhost:{port}"
    print(f"SignalSync dashboard running at {url}")
    print("Press Ctrl+C to stop the server.")
    webbrowser.open(url)
    HTTPServer(("127.0.0.1", port), DashboardHandler).serve_forever()