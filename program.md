# AutoACAP Research Program

This is an experiment to have an AI agent autonomously optimize ACAP applications on Axis camera hardware.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g., `mar17`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `src/c/app.c` — the C variant you can modify.
   - `src/rust/src/main.rs` — the Rust variant you can modify.
   - `evaluate.sh` — the fixed benchmark harness. Do not modify.
   - `deploy/` scripts — deployment infrastructure. Do not modify.
   - `benchmark/scripts/` — metric collection and mAP computation. Do not modify.
4. **Verify camera access**: `ssh root@192.168.1.33` — confirm the Q6358-LE is reachable.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment deploys to a real AXIS Q6358-LE camera (ARTPEC-9, aarch64). The benchmark harness runs for a **fixed 60-second benchmark window** after a 10-second warmup. You launch it as:

```bash
./evaluate.sh <c|rust> 192.168.1.33 <camera_password> > run.log 2>&1
```

**What you CAN do:**
- Modify `src/c/app.c` — the C person detection pipeline. Everything is fair game: buffer management, threading model, image pre-processing, memory layout, compiler hints, NEON intrinsics.
- Modify `src/rust/src/main.rs` — the Rust person detection pipeline. Same scope: allocators, concurrency patterns, unsafe FFI optimization, compile-time generics.
- Modify `src/c/Makefile` — compiler flags only (e.g., -O3, -march, LTO, PGO).
- Modify `src/rust/Cargo.toml` — optimization profiles, feature flags.

**What you CANNOT do:**
- Modify `evaluate.sh`, `deploy/` scripts, or `benchmark/scripts/`. They are read-only infrastructure.
- Modify the ground truth labels or benchmark video.
- Install new system packages on the camera.
- Change the ML model (ssd_mobilenet_v2_int8.tflite is fixed).

**The goal is simple: get the highest FPS while maintaining mAP >= 0.40.**

The 60-second benchmark window is fixed, so you don't need to worry about benchmark duration. Everything else is fair game: change the buffer management, threading model, pre-processing pipeline, memory layout, compiler flags — anything that makes inference faster while keeping accuracy above the gate.

**The first run**: Your very first run should always be to establish the baseline for each language, so run the C variant first unmodified, then the Rust variant unmodified.

## Output Format

The evaluate.sh script outputs a single tab-separated line:

```
fps	mAP	latency_p95	memory_mb	cpu_pct	binary_kb
15.2	0.62	45.3	128.5	32.1	256
```

Extract the key metrics from the log:
```bash
grep "^RESULT:" run.log
```

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 8 columns:

```
commit	lang	fps	mAP	latency_p95	memory_mb	cpu_pct	binary_kb	status	description
```

1. git commit hash (short, 7 chars)
2. language: `c` or `rust`
3. FPS achieved
4. mAP against ground truth
5. P95 latency in ms
6. Memory RSS in MB
7. CPU utilization %
8. Binary size in KB
9. status: `keep`, `discard`, or `crash`
10. short text description of what this experiment tried

Example:
```
commit	lang	fps	mAP	latency_p95	memory_mb	cpu_pct	binary_kb	status	description
a1b2c3d	c	14.8	0.62	52.3	128.5	34.2	245	keep	baseline C implementation
b2c3d4e	rust	14.2	0.61	55.1	95.2	31.8	312	keep	baseline Rust implementation
c3d4e5f	c	15.9	0.62	48.1	125.3	33.5	245	keep	arena allocator for per-frame buffers
d4e5f6g	rust	13.1	0.38	62.4	88.1	29.5	310	discard	aggressive NMS threshold (mAP below gate)
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g., `autoresearch/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Choose a language (C or Rust) and an optimization idea.
3. Modify the relevant source file with the experimental change.
4. Git commit.
5. Run the experiment: `./evaluate.sh <lang> 192.168.1.33 <pass> > run.log 2>&1`
6. Read out the results: `grep "^RESULT:" run.log`
7. If grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the error. Attempt a fix if trivial. If you can't fix after 3 attempts, give up on this idea.
8. Record the results in results.tsv (do NOT commit results.tsv — leave it untracked).
9. **Accuracy gate:** If mAP < 0.40, treat as a discard regardless of FPS.
10. If FPS improved (higher) and mAP >= 0.40, you "advance" the branch, keeping the commit.
11. If FPS is equal or worse, you `git reset` back to where you started.

## Language Rotation Strategy

- Spend 5 experiments optimizing the C variant
- Then 5 experiments optimizing the Rust variant
- Review which language is performing better
- Double down on the leader, but keep exploring the other
- Repeat

This rotation ensures both languages get attention, but the agent can shift focus to whichever is more promising.

## What to Explore (prioritized)

1. **Buffer management** — arena/pool allocators vs malloc/free (C) vs Box/Vec (Rust)
2. **Threading model** — single-threaded polling vs pipelined (capture → detect → post-process)
3. **Pre-processing** — manual NV12→RGB conversion vs VDO format flags
4. **Memory layout** — struct of arrays vs array of structs for detection results
5. **Compiler optimization** — LTO, PGO, -O3 vs -Os, target-specific flags (-march=armv8.2-a)
6. **SIMD** — NEON intrinsics for hot paths (NMS, image conversion)
7. **Unsafe FFI optimization** (Rust) — minimize C API crossing overhead
8. **Custom allocators** (Rust) — bump allocator for per-frame temporaries

## ACAP-Specific Constraints

- Camera: AXIS Q6358-LE (ARTPEC-9, 2GB RAM shared with OS, aarch64)
- DLPU: Chip ID 12, ~4.0 TOPS
- Model: ssd_mobilenet_v2_int8.tflite (quantized INT8)
- VDO stream: 640x480, NV12, 15 FPS source
- Must not interfere with camera's own video encoding pipeline
- Must not exceed ~256MB RSS (leave headroom for OS + other ACAPs)

## Crash Handling

- **Build failure:** Read compiler error, fix, retry (max 3 attempts). If unfixable, skip idea.
- **Camera unreachable:** Try SSH ping. If down, run `./deploy/reboot_camera.sh`. Wait 90 seconds. Retry once. If still down, stop and alert.
- **ACAP crash on startup:** Check syslog via SSH (`tail /var/log/syslog`). Fix if trivial, skip if fundamental.
- **Timeout (>3 min for evaluate.sh):** Kill and treat as crash. Likely OOM or DLPU hang.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the source files, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
