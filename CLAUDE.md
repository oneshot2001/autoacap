# Autoacap — CLAUDE.md

## Vault Memory
At session start, read these files in order:
1. ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/context.md
2. ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/decisions.md
3. ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/patterns.md
4. ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/bugs.md
5. ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/architecture.md
6. Last 3 files (by date) in ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/dev-log/
7. All files in ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/05-Knowledge/patterns/

Vault root: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain

## Auto-Capture Rules
During this session, track:
1. Every architectural decision (what, alternatives considered, why)
2. Every bug fixed (symptom, root cause, fix, prevention rule)
3. Every reusable pattern discovered (code snippet, when to use, where it applies)
4. Architecture changes (new routes, schema changes, data flow changes)

At session end, automatically write:
- Session log to: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/dev-log/YYYY-MM-DD-session-N.md
- Append new decisions to: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/decisions.md
- Append new bugs to: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/bugs.md
- Update if changed: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Second Brain/02-Projects/autoacap/architecture.md

## Session Log Format

Use this template for dev-log entries:

```markdown
---
date: YYYY-MM-DD
session: N
project: autoacap
tags: [dev-log]
---

# Dev Session — Autoacap — YYYY-MM-DD #N

## Summary
[1-2 sentence overview of what this session accomplished]

## What Was Built
- [Feature/fix 1]: [brief description]
- [Feature/fix 2]: [brief description]

## Decisions Made
### [Decision Title]
- **Options considered:** [option A], [option B], [option C]
- **Chosen:** [option]
- **Reasoning:** [why this option won]
- **Trade-offs:** [what we gave up]

## Bugs Encountered & Fixed
### [Bug Title]
- **Symptom:** [what you saw / error message]
- **Root cause:** [why it happened]
- **Fix:** [what resolved it]
- **Prevention:** [rule to avoid it in future]
- **Files changed:** [file paths]

## Patterns Discovered
### [Pattern Name]
- **When to use:** [description]
- **Reuse in:** [[Project1]], [[Project2]]

## Architecture Changes
- [Change description]: [before] -> [after]

## Reasoning Chains
[For complex decisions, document the full chain of reasoning that led to the outcome]

## Open Questions
- [ ] [Question that came up but wasn't resolved this session]

## Next Session Should
- [Top priority for next session]
- [Context that would be lost without writing it down]
```

## Project Details

### Stack
- C (ACAP Native SDK Docker — `axisecp/acap-native-sdk:12.9.0-aarch64-ubuntu24.04`)
- Rust (acap-rs + cargo-acap-build)
- Bash (benchmark harness, deploy scripts)
- Python (mAP computation, analysis notebook)
- Git (experiment tracking — autoresearch pattern)

### Build Commands
```bash
# C variant — build via Docker
cd src/c && docker build -t autoacap-c .

# Rust variant — build via cargo-acap-build
cd src/rust && cargo-acap-build

# Run benchmark (builds + deploys + benchmarks + scores)
./evaluate.sh <c|rust> 192.168.1.33 <camera_password>

# Analysis notebook
jupyter notebook analysis/analysis.ipynb
```

### Deploy
```bash
# Deploy .eap to camera
./deploy/deploy_eap.sh <c|rust> 192.168.1.33 <camera_password>

# Start/stop ACAP
./deploy/start_app.sh 192.168.1.33 <camera_password>
./deploy/stop_app.sh 192.168.1.33 <camera_password>

# Emergency recovery
./deploy/reboot_camera.sh 192.168.1.33 <camera_password>
```

### Quality Gates
1. C variant compiles without warnings (`-Wall -Werror`)
2. Rust variant compiles without warnings (`cargo clippy`)
3. .eap deploys and starts on Q6358-LE without crash
4. FPS ≥ baseline after optimization
5. mAP ≥ 0.40 accuracy gate
6. Git: committed to autoresearch branch

### Camera Details
- **Camera:** AXIS Q6358-LE (ARTPEC-9, aarch64, AXIS OS 12)
- **IP:** 192.168.1.33
- **DLPU:** Chip ID 12
- **Model:** SSD MobileNet V2 INT8 (.tflite)
- **Stream:** 640x480, NV12, 15 FPS

### Autoresearch Protocol
This project follows the Karpathy autoresearch pattern:
- `program.md` — agent instructions (human writes)
- `src/c/app.c` and `src/rust/src/main.rs` — agent modifies
- `evaluate.sh` — fixed benchmark harness (DO NOT MODIFY)
- `results.tsv` — experiment log (untracked by git)
- Git branch tip = best known configuration
- Successful experiments advance the branch; failures get reset
