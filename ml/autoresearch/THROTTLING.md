# Training throttling / kills — diagnosis & fix

## Symptoms observed
- Background training epochs ~53–65 s each; a short *foreground* probe was ~4.5 s/epoch
  (400k) — roughly a **3× slowdown** in the background.
- No fan, no heat, machine stays cool during the "slow" runs.
- Long background runs get **killed** after ~30–40 min (before finishing 50 epochs).
- A short *synchronous* run was **SIGKILLed (exit 137)** almost immediately, right after
  "resumed", before epoch 1.

## Evidence gathered
- `pmset -g therm`: **no thermal warning, no performance warning**; low-power-mode off. → not thermal.
- `top`: system shows **~30% idle CPU** while training crawls → process is **under-scheduled**,
  not starved of hardware.
- Process tree: `python ← caffeinate ← zsh ← claude.app` (PID 5849, Claude Code Bash tool).
  `nice` = 0 (not niced). So it's a **detached, non-interactive GUI-launched job**.
- Memory: `top` showed **~240 MB unused, ~7 GB memory compressor**, heavy swap counters. → the
  machine is **memory-starved**; loading the dataset OOM-kills the process (exit 137).

## Root causes (two, compounding)
1. **macOS background QoS / App Nap.** Claude Code's Bash tool auto-detaches long commands; macOS
   gives detached non-interactive jobs a low scheduling class → ~3× slower, cool, idle CPU free.
   (A foreground/synchronous command runs at full QoS — confirmed by the fast probe.)
2. **Memory pressure → OOM (SIGKILL/137).** Only ~240 MB free; `load_standardized` loads the FULL
   dataset (4.5M rows) into memory before subsampling, on top of Claude.app + other apps → OOM.

Neither is a PyTorch/model problem.

## Fixes (in order of effectiveness)
1. **Run in a real Terminal.app/iTerm window**, not via the assistant's Bash tool. Foreground,
   interactive → full QoS, no auto-detach, no ~3× penalty. This is the clean fix; the assistant
   can't launch a true-foreground long job through its tool (it auto-backgrounds).
2. **Free memory** before training: quit other heavy apps; the machine has little headroom
   (~240 MB free). Also make data loading lighter — load only a subsample, or stream — instead of
   loading all 4.5M rows then subsampling. (Avoids exit 137.)
3. If running through the assistant: **short synchronous chunks** under the 10-min Bash limit
   (e.g. `EPOCHS=8`) using the **resume** logic in `train_reparam.py` (loads `flow_reparam.pt`,
   continues, saves every improvement). Each chunk runs at full QoS; progress accumulates across
   chunks and survives kills.
4. Keep `caffeinate -dimsu` to stop idle-sleep when on battery (separate issue seen earlier when
   the Mac was unplugged and idle: it suspended the job entirely → ~930 s stalls).

## Current state
- Best reparam checkpoint (`models/flow_reparam.pt`) is at ~epoch 30, near-converged
  (val NLL plateaued ~0.858): smooth single flow, Rough 0.039, TailShape 0.12, high-z far tail
  near-exact (stress z=8 P(mu>5) 0.0099 vs sim 0.0101), weak only at z=2.
- `train_reparam.py` now supports `EPOCHS=<n>` env override + resume for safe chunked training.
