# ComWERA — Development Roadmap (next milestones)

This document is the execution plan for the next two milestones, written to be **self-contained**
so a fresh Claude Code session scoped to the ComWERA repo can resume without re-deriving context.
It covers **Performance Phase 3** (persistent per-GPU OpenMM contexts + a runner CV/burst hook) and
**Committor M4** (shooting-point / AIMMD augmentation). M4 depends on the runner hook from Phase 3,
so **do Phase 3 first**.

---

## 0. Current state (context recap)

**Branches / repos.** Development moved to `TeamSuman/ComWERA` (private). The `committor` branch
carries the full history: the performance refactor (Phases 1–2) *and* committor M1–M3.

**What is implemented (all CPU-tested, 87 tests; no GPU/MD/torch needed):**
- Performance Phase 1 — bug fixes + hot-path optimization (`cowera/features.py`, `metric.py`,
  `resampler.py`).
- Performance Phase 2 — in-memory CV history (`cowera/cv_history.py`); analysis decoupled from I/O
  in `metric.py` (`phase_from_projection`, `intensity_from_projections`, `project_new_frames`).
- Committor M1 — `cowera_committor/{featurizer,committor_model,committor_data,toy_systems}.py`.
- Committor M2 — `cowera_committor/committor_metric.py` (`CommittorDistances`: q-inference
  projection, continuous Δq phase, committor-augmented merge). **No change to `resampler.py`.**
- Committor M3 — `cowera_committor/{phase_manager,bootstrap,committor_train,committor_resampler}.py`;
  `run_cowera_committor.py`; `config_committor.yml`.

**Key design invariant.** The committor is a different progress coordinate feeding the *same*
phase→intensity machinery; `CommittorDistances` subclasses `Calculate_Distances` and is passed as
the resampler's `distance`. Preserve this — new work should not fork the resampler core.

**Run the tests:** `python -m pytest Scripts/tests/ -v`.

**Open validation (host-only, no GPU here):** the integrated in-memory resampler path, the
committor metric/resampler with real trajectories, and all multi-GPU performance work must be
validated on the simulation host (chignolin / Trp-cage; cf. `docs/REPRODUCING_THE_PAPER.md`).

---

## 1. Performance Phase 3 — persistent per-GPU contexts + runner CV/burst hook

### 1.1 Problem
`wepy/runners/openmm.py::OpenMMRunner.run_segment` builds a **fresh** `omma.Simulation` + CUDA
`Context` on **every walker every cycle** (recompiling kernels, re-uploading the system). The
process-per-task `TaskMapper` (`wepy/work_mapper/task_mapper.py`) also spins up a new `mp.Manager`
and forks N processes per cycle. For short resampling intervals this dominates wall time and starves
the GPUs. This is the single biggest multi-GPU inefficiency.

### 1.2 Approach
1. **Switch to persistent workers.** In `run_cowera.py` and `run_cowera_committor.py`, replace
   `TaskMapper(walker_task_type=OpenMMGPUWalkerTaskProcess, ...)` with
   `WorkerMapper(num_workers=n_gpu, worker_type=OpenMMGPUWorker, device_ids=gpu_ids, platform='CUDA')`
   (both exist: `wepy/work_mapper/mapper.py::WorkerMapper`, `wepy/runners/openmm.py::OpenMMGPUWorker`).
   Workers are long-lived (one per GPU) and consume tasks from a queue.
2. **Per-process Context cache** in `OpenMMRunner.run_segment`. Keep a *module-global* cache keyed by
   `(os.getpid(), platform_name, device_id, id(self.system))`. On first call in a worker, build the
   `Simulation`/`Context` once; on subsequent calls only:
   ```
   simulation.context.setState(walker.state.sim_state)
   simulation.context.setVelocitiesToTemperature(...)   # if a fresh integrator seed is desired
   simulation.step(segment_length)
   new_state = simulation.context.getState(**getState_kwargs)
   ```
   The cache MUST be created inside the worker process (after fork) — never build a CUDA context in
   the parent. Re-assign `simulation.reporters = [DCDReporter(walker_{i}.dcd, save_freq, append=...)]`
   each segment so per-walker DCDs still work.
   - Note on the integrator random seed: currently a new integrator is `copy()`ed per segment with
     seed 0. With a persistent context, reuse the cached integrator; reseed per segment via
     `integrator.setRandomNumberSeed(0)` to retain stochasticity. Verify reproducibility semantics.
3. **Trim state transfer.** `GET_STATE_KWARG_DEFAULTS` currently fetches forces + parameter
   derivatives every segment though reporters save only positions + box. Make the runner default to
   `getPositions=True, getVelocities=True, getEnergy=False, getForces=False, getParameterDerivatives=False`
   (or expose via config), cutting GPU→CPU transfer.
4. **Runner CV/burst hook** (the part M4 needs). Add two capabilities to `OpenMMRunner`:
   - **CV hook:** optional callback `cv_fn(positions) -> q` (or geometric CV) run on the saved frames
     inside the worker, returning the per-frame CV series attached to the new walker (e.g. in
     `OpenMMState._data['cv_series']`). This removes the resampler's per-cycle disk re-read in
     `CommittorDistances.project_new_frames` / `_load_projection` (the last disk seam). The committor
     `MLPCommittor` is picklable (pure numpy), so it can be shipped to workers; for the future
     GVP-GNN backend run inference on the worker's GPU.
   - **Burst hook:** `run_burst(state, n_steps, reverse=False) -> trajectory/states`. Reuses the
     cached context: `setState`; if `reverse`, negate velocities (time reversal); `step`; return the
     short trajectory. This is the primitive M4 uses for shooting points.

### 1.3 Files
- `Scripts/wepy/runners/openmm.py` — context cache in `run_segment`; trim `GET_STATE_KWARG_DEFAULTS`;
  add `cv_fn` plumbing and `run_burst`.
- `Scripts/run_cowera.py`, `Scripts/run_cowera_committor.py` — `TaskMapper` → `WorkerMapper`.
- (committor) `cowera_committor/committor_metric.py` — when a `cv_series` is present on the walker
  state, use it instead of loading the DCD in `project_new_frames` / `load_frames`.

### 1.4 Testing
- CPU smoke test on the OpenMM **Reference** platform with a tiny system (e.g. a few-particle
  harmonic system): assert the Context is built once across many `run_segment` calls (instrument the
  cache with a creation counter), and that `setState → step → getState` advances state correctly.
- Host benchmark: wall-clock per cycle, `TaskMapper` vs `WorkerMapper`, on chignolin; expect a large
  drop in per-segment overhead and higher GPU utilization.
- Regression: existing 87 tests stay green; rate/MFPT on chignolin unchanged within error.

### 1.5 Risks
- CUDA + fork: do not initialize CUDA in the parent; build contexts lazily in workers.
- Reporter/file handles with a persistent simulation (re-assign per segment; flush/close correctly).
- State round-trip fidelity (`setState` must restore positions+velocities+box exactly).

---

## 2. Committor M4 — shooting-point (AIMMD) augmentation

### 2.1 Goal
Add the only *dynamically rigorous* committor training signal: from configurations near `q≈0.5`,
launch short forward/backward bursts and count how many reach B vs A. This implements cold-start
**Phase 2** and drives the **histogram test** that promotes the model to **Phase 3 (mature)**.
The AIMMD loss is already implemented and gradient-checked in
`cowera_committor/committor_model.py` (the `shoot` data term); the `ShootingBuffer` already exists.

### 2.2 Approach (new `Scripts/cowera_committor/shooting.py`)
Pure, unit-testable helpers + a runner-backed driver:
1. `logit(q)` and `lorentzian_weights(q, gamma=...)` — selection weight `P ∝ 1/(logit(q)² + γ²)`,
   peaked at `q=0.5`. *(pure / testable)*
2. `select_shooting_points(frames, q, n_select, gamma, rng)` — sample shooting frames preferentially
   near the transition state. *(pure / testable)*
3. `harvest_reactive_fragment(q_series, a_cutoff, b_cutoff)` — given a walker's accumulated `q`
   series, return the index window from the last A-touch to the first B-touch (the reactive
   fragment). *(pure / testable)*
4. `run_shooting(runner, state, length, n_burst, boundary_fn, rng) -> (n_A, n_B, recycled)` — for one
   shooting point, launch `n_burst` forward + backward bursts via the **Phase 3 burst hook**;
   classify each endpoint with `boundary_fn` (q>b → B, q<a → A); return counts and any
   intermediate/B-hitting endpoints for recycling. *(needs runner; host-validated)*

### 2.3 Wiring into `committor_resampler.py` (Phase 2 only)
- Detect first **B-touch** per walker from the in-memory `q` history (reuse `CVHistory` / the
  per-walker `q` from `CommittorDistances`); extract the reactive fragment.
- On the retrain schedule while `phase == SHOOTING`: select shooting points, run bursts, add
  `(features(x_shoot), n_A, n_B, weight)` to `self.shooting` (`ShootingBuffer`), and **recycle** burst
  endpoints (B-hitting → `boundary_B`; intermediate → `semigroup`) so no MD is wasted.
- Compute `p_B` for a held-out set of `q≈0.5` candidates (via bursts) and pass it as
  `metrics['p_B']` to `phase_manager.record_and_maybe_advance` — the histogram-test path
  (`histogram_test_pass`) and Phase 2→3 advancement are already implemented.
- Loss weights for Phase 2 (`aimmd=2.0, semigroup=0.5, boundary=1.0`) are already returned by
  `BootstrapPhaseManager.loss_weights()`.

### 2.4 Files
- New: `Scripts/cowera_committor/shooting.py`; `Scripts/tests/test_committor_shooting.py`.
- Edit: `cowera_committor/committor_resampler.py` (Phase-2 hook); `cowera_committor/__init__.py`
  (export pure helpers); `run_burst` consumed from the Phase-3 runner.

### 2.5 Testing
- CPU (pure): `lorentzian_weights` peaks at `q=0.5`; `select_shooting_points` concentrates near 0.5;
  `harvest_reactive_fragment` returns the correct window; AIMMD loss already gradient-checked.
- Host: end-to-end bursts on the double well / alanine dipeptide; committor **histogram test** peaks
  at 0.5 ± 0.1; Phase 2→3 advancement fires; rate estimate stabilizes.

### 2.6 Cold-start Phase 3 (mature) — close-out after M4
Once the histogram test passes: full committor guidance with `merge_band=0` (already keyed off the
phase), sliding-window buffers (already implemented), and a **JSD-convergence** stop on the
`q`-histogram across cycles (add a small `jsd(p, q)` helper + a convergence check in the resampler).

---

## 3. Suggested execution order

1. Phase 3.1–3.3 (persistent workers + context cache + trimmed state) → benchmark on the host.
2. Phase 3.4 (CV hook) → switch `CommittorDistances` to consume `cv_series` (removes the disk seam).
3. Phase 3.4 (burst hook) → unblocks M4.
4. M4 (shooting.py + Phase-2 wiring) → host histogram test → Phase-3 mature close-out.
5. Then Stage 2 (GVP-GNN backend behind the `CommittorModel` interface) and benchmarks vs the
   published chignolin / Trp-cage numbers.

Keep the cadence used so far: small, CPU-tested commits; pure logic factored out and unit-tested;
integrated GPU/MD paths validated on the host before merging to the default branch.
