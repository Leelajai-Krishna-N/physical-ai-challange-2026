import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8001
WORKSPACE = Path(r"C:\physical-ai-challange-2026")

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>SO-101 Joint Controller</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <style>
    :root {
      --bg: #0f1418;
      --panel: #172128;
      --panel-2: #22323d;
      --text: #eff5f8;
      --muted: #8ea2ae;
      --accent: #57c7a6;
      --warn: #e6b85c;
      --shadow: rgba(0, 0, 0, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, #1e2931, transparent 30%),
        radial-gradient(circle at bottom right, #13282a, transparent 35%),
        var(--bg);
      color: var(--text);
      padding: 20px;
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 20px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 18px 40px var(--shadow);
    }
    h1, h2, p { margin: 0; }
    h1 { font-size: 28px; margin-bottom: 8px; }
    .sub { color: var(--muted); margin-bottom: 18px; }
    .joint-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(160px, 1fr));
      gap: 14px;
    }
    .joint-card {
      background: var(--panel);
      border-radius: 16px;
      padding: 14px;
      border: 1px solid rgba(255,255,255,0.06);
    }
    .joint-title {
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }
    .stack {
      display: grid;
      gap: 8px;
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 14px;
      padding: 14px 12px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      color: var(--text);
      background: var(--panel-2);
      transition: transform 0.04s ease, background 0.15s ease;
    }
    button:hover { background: #2c3f4d; }
    button:active { transform: translateY(1px); }
    .primary { background: #245848; }
    .primary:hover { background: #2a6754; }
    .warn { background: #675621; }
    .warn:hover { background: #78642a; }
    .status {
      margin-top: 16px;
      min-height: 24px;
      color: var(--accent);
      font-weight: 700;
    }
    .actions, .meta { display: grid; gap: 12px; }
    .row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    input[type="range"] { width: 100%; }
    .pill {
      display: inline-block;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      color: var(--muted);
      margin-right: 8px;
      margin-bottom: 8px;
      font-size: 13px;
    }
    .kbd {
      font-family: Consolas, monospace;
      background: rgba(255,255,255,0.08);
      border-radius: 8px;
      padding: 2px 6px;
    }
    @media (max-width: 960px) {
      .wrap { grid-template-columns: 1fr; }
      .joint-grid { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>SO-101 Joint Controller</h1>
      <p class="sub">Direct joint steps only. No analog drift, no IK wobble, just explicit commands.</p>

      <div class="joint-grid">
        <div class="joint-card">
          <div class="joint-title">Shoulder Pan</div>
          <div class="stack">
            <button data-joint="0" data-dir="1" class="primary">Pan +</button>
            <button data-joint="0" data-dir="-1">Pan -</button>
          </div>
        </div>
        <div class="joint-card">
          <div class="joint-title">Shoulder Lift</div>
          <div class="stack">
            <button data-joint="1" data-dir="1" class="primary">Lift +</button>
            <button data-joint="1" data-dir="-1">Lift -</button>
          </div>
        </div>
        <div class="joint-card">
          <div class="joint-title">Elbow Flex</div>
          <div class="stack">
            <button data-joint="2" data-dir="1" class="primary">Elbow +</button>
            <button data-joint="2" data-dir="-1">Elbow -</button>
          </div>
        </div>
        <div class="joint-card">
          <div class="joint-title">Wrist Flex</div>
          <div class="stack">
            <button data-joint="3" data-dir="1" class="primary">Wrist Flex +</button>
            <button data-joint="3" data-dir="-1">Wrist Flex -</button>
          </div>
        </div>
        <div class="joint-card">
          <div class="joint-title">Wrist Roll</div>
          <div class="stack">
            <button data-joint="4" data-dir="1" class="primary">Roll +</button>
            <button data-joint="4" data-dir="-1">Roll -</button>
          </div>
        </div>
        <div class="joint-card">
          <div class="joint-title">Gripper</div>
          <div class="stack">
            <button id="openGrip" class="primary">Open</button>
            <button id="closeGrip" class="warn">Close</button>
          </div>
        </div>
      </div>

      <div class="meta" style="margin-top: 18px;">
        <div>
          <div class="joint-title">Joint Step Size</div>
          <input id="stepSize" type="range" min="0.02" max="0.30" step="0.01" value="0.08" />
          <div id="stepLabel" class="status" style="margin-top:8px;">Current joint step: 0.08 rad</div>
        </div>
        <div>
          <span class="pill"><span class="kbd">1/2</span> pan +/-</span>
          <span class="pill"><span class="kbd">3/4</span> lift +/-</span>
          <span class="pill"><span class="kbd">5/6</span> elbow +/-</span>
          <span class="pill"><span class="kbd">7/8</span> wrist flex +/-</span>
          <span class="pill"><span class="kbd">9/0</span> roll +/-</span>
          <span class="pill"><span class="kbd">O/C</span> gripper</span>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2 style="margin-bottom: 14px;">Actions</h2>
      <div class="actions">
        <div class="row">
          <button id="homeBtn">Home Pose</button>
          <button id="refreshBtn">Refresh UI</button>
        </div>
        <div class="row">
          <button id="fineBtn">Fine 0.04</button>
          <button id="normalBtn">Normal 0.08</button>
        </div>
      </div>
      <div id="status" class="status">Ready.</div>
    </section>
  </div>

  <script>
    const stepSize = document.getElementById('stepSize');
    const stepLabel = document.getElementById('stepLabel');
    const status = document.getElementById('status');

    function updateStepLabel() {
      stepLabel.textContent = `Current joint step: ${Number(stepSize.value).toFixed(2)} rad`;
    }

    async function send(payload, successText) {
      try {
        await fetch('/command', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        if (successText) status.textContent = successText;
      } catch (err) {
        status.textContent = `Failed: ${err}`;
      }
    }

    function sendJointStep(index, dir) {
      const step = Number(stepSize.value) * Number(dir);
      send({ kind: 'joint_step', joint: Number(index), delta: step }, `Joint ${Number(index) + 1} moved by ${step.toFixed(2)} rad`);
    }

    document.querySelectorAll('[data-joint]').forEach((button) => {
      button.addEventListener('click', () => sendJointStep(button.dataset.joint, button.dataset.dir));
    });

    document.getElementById('openGrip').addEventListener('click', () => send({kind: 'gripper', target: 1.74}, 'Opening gripper'));
    document.getElementById('closeGrip').addEventListener('click', () => send({kind: 'gripper', target: 0.0}, 'Closing gripper'));
    document.getElementById('homeBtn').addEventListener('click', () => send({kind: 'home'}, 'Returning to home pose'));
    document.getElementById('refreshBtn').addEventListener('click', () => location.reload());
    document.getElementById('fineBtn').addEventListener('click', () => { stepSize.value = 0.04; updateStepLabel(); status.textContent = 'Fine step enabled'; });
    document.getElementById('normalBtn').addEventListener('click', () => { stepSize.value = 0.08; updateStepLabel(); status.textContent = 'Normal step enabled'; });

    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      const key = e.key.toLowerCase();
      const map = {
        '1': [0, -1], '2': [0, 1],
        '3': [1, -1], '4': [1, 1],
        '5': [2, -1], '6': [2, 1],
        '7': [3, -1], '8': [3, 1],
        '9': [4, -1], '0': [4, 1],
      };
      if (map[key]) sendJointStep(map[key][0], map[key][1]);
      if (key === 'o') send({kind: 'gripper', target: 1.74}, 'Opening gripper');
      if (key === 'c') send({kind: 'gripper', target: 0.0}, 'Closing gripper');
      if (key === 'h') send({kind: 'home'}, 'Returning to home pose');
    });

    stepSize.addEventListener('input', updateStepLabel);
    updateStepLabel();
  </script>
</body>
</html>
"""


relay_process = None
relay_lock = threading.Lock()


def ensure_relay():
    global relay_process
    with relay_lock:
        if relay_process is not None and relay_process.poll() is None:
            return relay_process
        relay_process = subprocess.Popen(
            [
                "docker", "exec", "-i",
                "lerobot_hackathon_env",
                "bash", "-lc",
                (
                    "cd /home/hacker/workspace && "
                    "source /opt/ros/humble/setup.bash && "
                    "source install/setup.bash && "
                    "python3 src/so101_mujoco/scripts/so101_virtual_joystick_relay.py"
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(WORKSPACE),
        )
        line = relay_process.stdout.readline().strip()
        if "ready" not in line.lower():
            raise RuntimeError(f"relay failed to start: {line}")
        return relay_process


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/command":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length).decode("utf-8")
        proc = ensure_relay()
        assert proc.stdin is not None
        proc.stdin.write(data + "\n")
        proc.stdin.flush()
        self.send_response(204)
        self.end_headers()

    def log_message(self, format, *args):
        return


def main():
    ensure_relay()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Joint controller UI: http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
