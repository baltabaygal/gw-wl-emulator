# Training slowdown / kills — diagnosis & fix

## RESOLVED — root cause was a CODE bug, not the environment
`train_reparam.py` evaluated the **entire 4.3M-row validation set in one unbatched MPS
forward every epoch** → ~34–60 s/epoch. `train_ar.py` *batches* its validation, so it was
fast. My earlier "foreground 4.5 s vs background 55 s" comparison was **invalid**: it compared
*different code* (train_ar batched-val probe vs train_reparam unbatched-val run), not
foreground vs background. Evidence this is the real cause:
- `log show … jetsam` = **no OOM events** (ruled out memory-kill);
- the user's run in a **real Terminal was also ~60 s/epoch** (rules out the assistant's
  backgrounding / QoS);
- **subsampling the val to 100k → 2 s/epoch** (60s→2s) — fix confirmed.
So the QoS / App-Nap / OOM theories below were **wrong**; kept only as a record of what was
checked. The earlier on-battery idle-suspension (~930 s stalls) was a separate, real issue.

---
## (superseded) earlier hypotheses — kept for the record

Stated as **observations** (measured) vs **explanations** (inferred, with confidence).

## Observations (well supported)
- **Not thermal.** `pmset -g therm`: no thermal/performance warning; low-power-mode off; cool, no fan.
- **Foreground vs background throughput differs ~3×.** A short *synchronous* probe ran ~4.5 s/epoch
  (400k rows); long *backgrounded* runs ran ~53–65 s/epoch. Same machine, same code.
- **Long/background runs get terminated** (exit 137 = SIGKILL; one earlier 144) before finishing.
- **System had idle CPU** (~30%) during the slow runs (`top`).
- **No OS OOM events.** `log show --last 2h` for `jetsam`/`memorystatus`/`lowswap` returned
  **nothing**; `memory_pressure` reports **~54% free**. (The `top` "~240 MB unused" was misleading —
  macOS keeps RAM full with *reclaimable* compressor/cache.)

## Explanations (inferred — confidence noted)
1. **Execution environment is the dominant factor** (high confidence on *effect*, mechanism
   unverified). Long commands launched via Claude Code's Bash tool are auto-detached and run ~3×
   slower than equivalent foreground runs, and are terminated after a while. The behavior is
   *consistent with* reduced scheduling priority / background QoS (App Nap) on detached
   GUI-launched processes, **but the exact scheduling mechanism has not been directly measured.**
   The terminations are **not** OS OOM (no jetsam logs) → most likely the execution environment
   itself ending long/detached jobs.
   - Caveat: idle CPU + slow ≠ proof of QoS; it can also reflect page faults, compressed-memory
     waits, I/O, or sync stalls. Those look less likely here but aren't excluded.
2. **Memory pressure — possible secondary, NOT confirmed.** `top` showed little "unused" RAM and a
   large compressor, and `load_standardized` eagerly loads all ~4.5M rows before subsampling. But
   no jetsam events were logged and memory_pressure shows 54% free, so memory is **not** the proven
   cause of the kills. Treat as a defensive concern, not a diagnosis.

## Fixes
1. **Run training in a real Terminal.app / iTerm window** (not via the assistant). Foreground +
   interactive → full QoS, no auto-detach, no ~3× penalty, no tool-imposed termination. The
   assistant's Bash tool auto-backgrounds long commands, so it *cannot* hold a true-foreground
   long job — this is the clean fix. Exact command at the bottom.
2. **Lazy-load / subsample the dataset at read time** (elevated to equal priority). Loading only the
   needed subset instead of all 4.5M rows cuts peak memory by several GB and removes that failure
   class *regardless of where it runs*. (`load_standardized` currently loads everything then
   subsamples.)
3. **Through the assistant: short synchronous chunks** under the 10-min Bash limit (`EPOCHS=8`)
   using the **resume** logic in `train_reparam.py` (loads `flow_reparam.pt`, continues, saves on
   every improvement). Each chunk runs at full QoS; progress accumulates and survives terminations.
4. **`caffeinate -dimsu`** to prevent idle-sleep on battery (a separate, earlier issue: unplugged +
   idle suspended the job entirely → ~930 s stalls).

## Run-it-yourself command (full speed)
```bash
cd /Users/baltabay/Desktop/gw-wl-emulator-ar
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  /Users/baltabay/miniforge3/envs/test/bin/python -u ml/autoresearch/train_reparam.py
```

## Current state
- Best reparam checkpoint (`models/flow_reparam.pt`) ≈ epoch 30, near-converged (val NLL plateaued
  ~0.858): smooth single flow, Rough 0.039, TailShape 0.12; high-z far tail near-exact
  (stress z=8 P(mu>5) 0.0099 vs sim 0.0101), weak only at z=2.
- `train_reparam.py` supports `EPOCHS=<n>` env override + resume for safe chunked training.
