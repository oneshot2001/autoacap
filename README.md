# AutoACAP

Autonomous AI-driven optimization of ACAP applications on Axis camera hardware, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

An AI agent rewrites person detection ACAPs in C and Rust, deploys each variant to a real AXIS Q6358-LE camera (ARTPEC-9), benchmarks head-to-head, and ratchets toward the fastest implementation — overnight, unattended.

## How It Works

The repo has three layers:

- **`program.md`** — Instructions for the AI agent. The human writes this.
- **`src/c/app.c`** and **`src/rust/src/main.rs`** — The agent modifies these.
- **`evaluate.sh`** — Fixed benchmark harness. Builds, deploys, benchmarks, scores. Never modified.

Each experiment:
1. Agent proposes a change to `app.c` or `main.rs`
2. Git commits the change
3. `evaluate.sh` builds, deploys to camera, runs 60s benchmark
4. If FPS improved (and mAP >= 0.40) → keep commit
5. If FPS same or worse → `git reset`
6. Log to `results.tsv`, repeat forever

## Quick Start

```bash
# 1. Verify camera access
ssh root@192.168.1.33

# 2. Pull ACAP Native SDK Docker image
docker pull axisecp/acap-native-sdk:12.9.0-aarch64-ubuntu24.04

# 3. Validate model on camera DLPU
# (copy model to camera and test with larod)

# 4. Capture benchmark ground truth
# (see benchmark/scripts/capture_benchmark.sh)

# 5. Run a manual experiment
./evaluate.sh c 192.168.1.33 <password>

# 6. Start autonomous research
# Point Claude Code at this repo:
# "Read program.md and let's kick off a new experiment!"
```

## Project Structure

```
autoacap/
├── program.md              # Agent instructions (human writes)
├── evaluate.sh             # Fixed benchmark harness (DO NOT MODIFY)
├── results.tsv             # Experiment log (untracked)
├── src/
│   ├── c/                  # C variant
│   └── rust/               # Rust variant
├── benchmark/              # Video, ground truth, scripts
├── models/                 # .tflite model files
├── deploy/                 # Camera deploy/start/stop scripts
└── analysis/               # Post-run Jupyter notebook
```

## Target Hardware

- **Camera:** AXIS Q6358-LE (ARTPEC-9, aarch64, AXIS OS 12)
- **IP:** 192.168.1.33
- **Model:** SSD MobileNet V2 INT8 (.tflite, DLPU Chip ID 12)
- **Stream:** 640x480, NV12, 15 FPS

## Metric

- **Optimize:** FPS (higher is better)
- **Gate:** mAP >= 0.40 (discard if below)
- **Track:** latency_p95, memory_mb, cpu_pct, binary_kb

## License

MIT
