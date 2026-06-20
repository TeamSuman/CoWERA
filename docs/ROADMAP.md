# CoWERA — Development Roadmap

Forward-looking plan for the **CoWERA core method and its multi-GPU performance**. This is the
handoff document for resuming work after the current pause. Detailed bug/optimization analysis lives
in [`ANALYSIS_AND_OPTIMIZATION.md`](ANALYSIS_AND_OPTIMIZATION.md); benchmark reproduction in
[`REPRODUCING_THE_PAPER.md`](REPRODUCING_THE_PAPER.md).

> **Scope note.** Committor-based resampling (a learned `q(x)` as the clone/merge criterion) has been
> spun out into a separate project, **ComWERA**, and is no longer developed here. This repository and
> the `devel` branch focus on the published coherence-based method and its performance.

---

## Status snapshot

| Item | Status |
|---|---|
| Phase 1 — correctness bug fixes + hot-path optimization | ✅ done (`devel`, PR #2) |
| Phase 2 — in-memory CV history (decoupled analysis/IO) | ✅ done (`devel`, PR #2) |
| Phase 2 — host validation (integrated resampler path) | ⬜ pending (GPU host) |
| Phase 3 — persistent per-GPU OpenMM contexts | ⬜ next |
| Phase 4 — algorithmic accelerations | ⬜ future |

CPU test suite: **32 tests** (`python -m pytest Scripts/tests/ -v`), pure NumPy, no GPU/MD stack.

---

## Current state (what is done)

- **Phase 1** — fixed the changepoint operator-precedence bug, the divide-by-zero/NaN paths
  (`scale_phases` / `scale_weights` / `compute_intensity`), empty-segment phase, the broken
  feature fallbacks, and unified the RNG. Cached native contacts + RMSD references; removed the
  per-pair MDAnalysis Universe construction; vectorized the Q core (~150×); parallelized the phase
  loop. Files: `Scripts/cowera/{features,metric,resampler}.py`.
- **Phase 2** — `Scripts/cowera/cv_history.py` (`CVHistory`) keeps each walker's CV series in memory,
  updated incrementally with clone/merge reindexing and warp-reset detection; `metric.py` analysis is
  decoupled from I/O (`phase_from_projection`, `intensity_from_projections`, `project_new_frames`);
  the resampler uses the in-memory path by default (`use_cv_history=True`, legacy path preserved).

**Open validation:** the integrated in-memory resampler path needs a host run (no GPU/OpenMM in CI).
Validate before merging `devel` → `main`: a short chignolin run with `use_cv_history` on vs off,
comparing `Info_*.txt` and the rate/MFPT against `docs/REPRODUCING_THE_PAPER.md` targets.

---

## Phase 3 — Persistent per-GPU OpenMM contexts (next)

**Problem.** `wepy/runners/openmm.py::OpenMMRunner.run_segment` builds a fresh `omma.Simulation` +
CUDA `Context` on **every walker every cycle** (recompiling kernels, re-uploading the system), and
the process-per-task `TaskMapper` forks N processes + a new `mp.Manager` per cycle. For short
resampling intervals this dominates wall time and starves the GPUs — the single biggest multi-GPU
inefficiency.

**Approach.**
1. **Persistent workers.** In `run_cowera.py`, replace `TaskMapper(... OpenMMGPUWalkerTaskProcess)`
   with `WorkerMapper(num_workers=n_gpu, worker_type=OpenMMGPUWorker, device_ids=gpu_ids,
   platform='CUDA')` (both already exist in `wepy`). One long-lived process per GPU.
2. **Per-process Context cache** in `run_segment`, keyed by `(os.getpid(), platform, device_id,
   id(self.system))`. Build the `Simulation`/`Context` once per worker; thereafter only
   `context.setState(...) → integrator.step(...) → context.getState(...)`. Build the context
   **inside the worker** (after fork) — never create a CUDA context in the parent. Re-assign the
   per-walker `DCDReporter` each segment. Re-seed the integrator per segment to retain stochasticity.
3. **Trim state transfer.** Default `GET_STATE_KWARG_DEFAULTS` to positions + box (+ velocities)
   only; drop forces / parameter derivatives (reporters save nothing else), cutting GPU→CPU transfer.

**Files:** `Scripts/wepy/runners/openmm.py` (context cache; trimmed `getState`), `Scripts/run_cowera.py`
(mapper swap).

**Testing.** CPU smoke test on the OpenMM **Reference** platform with a tiny system: assert the
context is constructed once across many `run_segment` calls (instrument with a creation counter) and
that `setState → step → getState` advances correctly. Host benchmark: per-cycle wall time and GPU
utilization, `TaskMapper` vs `WorkerMapper`, on chignolin. Regression: 32 tests stay green; chignolin
rate/MFPT unchanged within error.

**Risks.** CUDA + fork interactions (lazy context creation in workers only); reporter/file-handle
management with a persistent simulation; exact `setState` round-trip fidelity
(positions + velocities + box).

---

## Phase 4 — Algorithmic accelerations (future)

- **GPU-inline CV reporting** via an OpenMM `CustomCVForce` (or a CV-reporting force) so the progress
  coordinate is produced **during** propagation — removing the CPU recompute + per-segment DCD read
  in the analysis path entirely. (Pairs naturally with the Phase-3 runner if a CV hook is added.)
- **Online/incremental changepoint detection** to replace per-cycle `ruptures.KernelCPD` (≈O(T²));
  only the most recent changepoint matters, so a streaming detector or bounded window suffices
  (the `CVHistory.window` already bounds growth as a stopgap).
- **Lazy distance matrix** — only the low-intensity merge candidate's distances feed `merge_mask`;
  compute RMSD on demand (or a cheap CV-space prefilter, then RMSD on borderline pairs) instead of
  the full O(N²) matrix every cycle.
- **CPU/GPU overlap** — pipeline cycle-`k` CPU resampling analysis with cycle-`k+1` GPU propagation.
- **Probabilistic merge survivor** — re-enable `P(keep i) = w_i/(w_i + w_j)` (the suppressed block in
  `resampler.decide`) as an option for strict WE statistical exactness.

---

## Validation & benchmarking track

- Land a small host benchmark harness reporting per-cycle wall time, sampling vs overhead split, and
  GPU utilization (the runner already records segment split times).
- After Phase 3, confirm the chignolin / Trp-cage kinetics still match Tables II–III, then merge
  `devel` → `main`.

---

## How to resume

1. Start from `devel` (PR #2); run `python -m pytest Scripts/tests/ -v` (expect 32 passing).
2. Implement Phase 3 §1–§3 above; add the Reference-platform context-reuse smoke test.
3. Validate + benchmark on the GPU host; then proceed to Phase 4 items as prioritized.

Keep the established cadence: small, CPU-tested commits; pure logic factored out and unit-tested;
GPU/MD-integrated paths validated on the host before merging to `main`.
