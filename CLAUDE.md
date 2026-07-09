# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It doubles as the **living plan** for the current effort: optimize CoWERA for general HPC, then
benchmark it against WESTPA. Keep the roadmap sections below revised as work lands.

---

## What this is

CoWERA (Coherence-based Weighted Ensemble Resampling Algorithm) is a **binless weighted-ensemble (WE)**
resampler for rare-event kinetics (protein folding/unfolding MFPTs and rate constants) from GPU MD.
It implements Shahid, Maity & Chakrabarty, *J. Chem. Phys.* **164**, 134111 (2026). Dynamics run on
**OpenMM**; the framework is a **vendored fork of `wepy`** under `Scripts/wepy/`. The manuscript lives
in the sibling `../CoWERA_manuscript/` (`manuscript.tex`, `supp.tex`).

## Environment & commands

Run everything **from the repository root** — `run_cowera.py` appends `Scripts/` and `Systems/` to
`sys.path` and resolves config paths relative to the CWD.

- **Env:** `conda env create -f env/environment.yml -n cowera && conda activate cowera`
  (Python 3.7, OpenMM 7.5.1, cudatoolkit 11.7, mdtraj, MDAnalysis 2.0.0, ruptures, joblib, eliot, h5py).
- **Run:** `python ./Scripts/run_cowera.py --config ./Systems/<name>/config.yml` (`--restart` to resume).
- **CPU-only:** set `platform: "CPU"`, `gpu_ids: []`, `n_workers: <cores>` in the config.
- **Cluster:** `Scripts/slurm/cowera.sbatch` (SLURM, incl. `--array` replicas) and
  `Scripts/sample_job_multirun.sh` (PBS); both handle CUDA MPS setup/teardown. See
  `docs/REPRODUCING_THE_PAPER.md` §3.1.
- **Fast tests (pure numpy, no MD stack):** `python -m pytest Scripts/tests/ -q`.
  **Host/MD tests:** `python -m pytest Scripts/tests/ -m integration` (needs OpenMM/mdtraj/ruptures).
- **Analysis:** `Analysis/WE_analysis.ipynb` (MFPT via Hill relation from `wepy.results.h5`).

## Architecture

Two layers — know which you're editing:
1. **`Scripts/cowera/`** — the CoWERA method (the science). Almost all algorithm work is here.
2. **`Scripts/wepy/`** — vendored `wepy` WE framework (OpenMM runners, work mappers, HDF5/dashboard
   reporters, clone/merge records). Treat as a dependency, but the runner/mapper are edited for perf.

`Scripts/run_cowera.py` is the monolithic entry point wiring everything; the WE cycle loop
(propagate → warp → resample → report) is `cowera/sim_manager.py::Manager`.

Key `cowera` modules:
- `features.py` — progress-coordinate CVs `best_hummer_q` (Q) and `RMSD_Backbone`; pure-numpy cores
  (`compute_q`) + file-keyed caches; mdtraj lazy-imported. Fallback-to-init counter for CV-read failures.
- `metric.py` — `Calculate_Distances`: projection, pairwise distance matrix, coherence intensity
  (phase → scaled phase → intensity; changepoints via `ruptures`). Pure cores are the unit-tested part.
- `resampler.py` — `CoWERAResampler`: `decide()` clone/merge to maximize variation (greedy or
  probabilistic); orchestrates per-cycle distance/intensity in `get_dist()`.
- `cv_history.py` — `CVHistory`: in-memory per-walker CV series, incremental update, clone/merge
  reindex, warp reset.
- `warper.py` — `TargetBC`: warp (recycle) a walker reaching the target within `d_warped`.
- `file_resampler.py` — mirrors clone/merge onto per-walker `walker_{i}.dcd` files.

**Adding a system:** `Systems/<name>/` with structure files (`.gro`+`.top` or `.pdb`+`.prmtop`), a
`config.yml` (copy `Scripts/config_template.yml`), and `system.py` defining
`make_system(top, temp) -> (system, integrator)`. Loaded dynamically via `importlib`.

## Config reference (fields added during the HPC overhaul)

All optional, back-compatible defaults preserve prior behaviour:
`platform` (CUDA|OpenCL|CPU|Reference), `platform_kwargs`, `n_workers`, `persistent_workers`,
`seed`, `deterministic_dynamics`, `checkpoint_freq`, `restart`, `scratch_dir`, `getstate_kwargs`,
`history_window`, `bin_increase_factor`/`bin_decrease_factor` (D-1), `i0_mode` (projection|uniform),
`use_phase` (false ⇒ targeted-WE baseline). See `Scripts/config_template.yml` for docs.

---

## ROADMAP A — Optimize CoWERA for general HPC

Goal: maximal GPU/CPU utilization and minimal wall-clock/GPU-hours per converged kinetic estimate, so
the WESTPA comparison (Roadmap B) reflects the **algorithm**, not implementation immaturity. Aggressive
refactoring of `Scripts/wepy/runners/openmm.py`, the work mapper, and `cowera/{metric,resampler}.py`
is in scope. **Profile before refactoring — do not optimize blind.**

**Status of prior work (branch `feature/hpc-cluster-enhancements`, 12 commits):** platform switch,
restart/checkpoint, SLURM/PBS scripts, scoped warnings + CV-fallback counter, seed plumbing, trimmed
`getState`, incremental projection (fixed the O(T) re-projection no-op), forwarded bin factors, **opt-in**
persistent per-GPU cached Context, `scratch_dir` staging, joblib CPU-alloc awareness, doc reconciliation,
`i0_mode`/`use_phase` baselines. 52 pure-numpy tests pass; OpenMM paths have `-m integration` tests.

### P0 — Profile first (prerequisite) — ✅ LANDED (commit 29f9819)
Harness built (behind config `profile: false`): `cowera/timing_reporter.py` (per-cycle wall-clock
breakdown → `<output>/profile.csv`, consuming the sim-manager timings + resampler `_last_subtimings`),
`tools/gpu_monitor.py` (GPU util/mem sampler), `tools/bench_run.py` (ranks the buckets + GPU-idle %).
6 unit tests. **Remaining (host):** run on the GPU host for chignolin (short ΔT = 1000 steps — worst
case) and Trp-cage, record the baseline, and let the ranked breakdown confirm the P1–P5 order:
```
python Scripts/tools/gpu_monitor.py --out gpu_util.csv &   # then run with profile: true
python Scripts/run_cowera.py --config ./Systems/chignolin/config.yml
python Scripts/tools/bench_run.py --profile-csv <output>/profile.csv --gpu-csv gpu_util.csv
```

### P1 — Make persistent workers + cached Context the default (biggest win)
- Validate the existing opt-in path on the host; make `persistent_workers` the default for GPU. Removes
  per-walker/per-cycle process fork + OpenMM Context rebuild (kernel recompile + system re-upload) — the
  dominant overhead for short ΔT. Confirm no per-cycle `mp.Manager` recreation (reuse the worker pool).
- Risks: CUDA+fork (build Context only inside workers), reporter/file-handle lifetime, `setState`↔
  `getState` round-trip fidelity (positions+velocities+box). Escape hatch stays for one release.

### P2 — Eliminate the per-cycle disk round-trip (runner-inline CV)
- Today `metric.project_new_frames` still `md.load`s the full DCD each cycle (only projection is
  incremental). Compute the CV (Q / backbone RMSD) **during/after propagation in the runner** and return
  it with the walker state, so the resampler never reads DCDs for analysis. Options: OpenMM `RMSDForce`
  / a `CustomCVForce` for Q reported per segment, or an in-runner projection of the final state appended
  to an in-memory CV stream. Decouples analysis from trajectory-file bookkeeping; DCDs become output-only
  (write async / less often). Removes the last O(T) term from the analysis path.

### P3 — Sub-quadratic changepoint + lazy distance matrix
- Replace per-cycle `ruptures.KernelCPD` over the full history (~O(T²)) with an online/streaming detector
  or a bounded window (only the most recent changepoint matters) → O(window). `CVHistory.window` already
  caps growth as a stopgap.
- Lazy distance matrix: only the low-intensity merge candidate needs distances; CV-space prefilter, then
  RMSD on borderline pairs → O(N²) → ~O(N·k).

### P4 — CPU/GPU overlap (hardest; last)
- Double-buffer so cycle-k CPU resampling analysis overlaps cycle-(k+1) GPU propagation → hide CPU work
  behind GPU work, pushing GPU utilization toward saturation for short ΔT. Careful: resampling
  reorganizes walkers, so overlap the analysis/reporting, not the clone/merge decision.

### P5 — Vectorize resampler + trim serialization (profile-driven)
- The `decide()` while-loop is Python-level; vectorize for larger walker counts. With persistent workers
  only minimal state crosses the process boundary (positions+velocities — Phase 2.1 already trims).

### Utilization / scaling targets
- Strong scaling: fixed walkers, vary GPUs (1/2/4) → near-linear wall-clock drop until walkers/GPU
  saturate. MPS: map the walkers-per-GPU vs throughput curve. Target GPU utilization > ~80% on
  chignolin short ΔT (likely low today due to per-cycle overhead).

---

## ROADMAP B — WESTPA vs CoWERA efficiency comparison

Goal: on the **same HPC setup**, show CoWERA reaches a converged kinetic estimate with less cost than
WESTPA. Do this only **after** Roadmap A, so the comparison isn't confounded by CoWERA's implementation
overhead. Report **both** algorithmic and computational efficiency so the algorithm's contribution is
separable.

### B0 — Metrics & fairness
- **Primary:** GPU-hours (or wall-clock on identical hardware) to a converged rate/MFPT within a target
  confidence interval. **Secondary:** aggregate MD time to convergence (algorithmic efficiency),
  run-to-run variance vs cost, time-to-first-reactive-event τ_FRE, GPU utilization, strong/weak scaling.
- **Fairness controls:** identical system + force field + engine (OpenMM) + integrator/timestep/T,
  identical state definitions (folded/unfolded cutoffs from `supp.tex`), same hardware/node, matched
  walker counts and resampling interval where the paradigms allow, identical analysis (Hill relation +
  block averaging + CIs), same #replicas (≥5) and wall-time budget, pinned/locked environments.

### B1 — WESTPA setup (competent, not a strawman)
- WESTPA 2.0 with the **OpenMM propagator**, chignolin implicit-solvent matching
  `Systems/chignolin/system.py`; folding + unfolding; states from `supp.tex` (RMSD cutoffs).
- Baselines: (a) traditional binned WE (Huber–Kim, the fixed RMSD bins in `supp.tex` §"WE Simulation
  Details"), (b) WESTPA **MAB** (adaptive binning — the strong modern baseline). Optionally the
  targeted-WE baseline (now reproducible in CoWERA via `use_phase: false`).
- Use WESTPA's recommended work manager (ZMQ) for the node so it is not I/O/overhead-bound unfairly.

### B2 — Run matrix
- Systems: chignolin (primary — cheap, many replicas), Trp-cage (secondary — the hard case).
- Methods: CoWERA (optimized) · WESTPA-binned · WESTPA-MAB · [targeted-WE].
- Hardware: one GPU node for the head-to-head; plus a 1/2/4-GPU scaling sweep. ≥5 replicas each.

### B3 — Analysis (identical for both)
- Same convergence criterion (run-to-run variance threshold) + block averaging for rate/MFPT + CIs
  across replicas (as in the paper). Compute GPU-hours-to-converged-CI, aggregate-MD-time-to-converged,
  variance-vs-cost, τ_FRE distributions, GPU utilization, scaling curves; report with error bars +
  significance tests.

### B4 — Deliverable
- A reproducible `benchmarks/` subtree: configs, SLURM/PBS scripts, env locks, analysis notebooks, raw
  + processed data, and tables/plots in the paper's style. This becomes the evidence for the efficiency
  claim.

### Threats to validity
- WESTPA is mature (persistent workers, HDF5, ZMQ/MPI). A wall-clock comparison against an unoptimized
  CoWERA measures implementation maturity, not the algorithm — hence Roadmap A first, and always report
  algorithmic (aggregate MD time) alongside computational (GPU-hours) efficiency.

---

## Continuation notes

- **Execution environments (set by the user).** Remote **HPC cluster with GPUs** = all profiling, GPU
  runs, OpenMM validation, and large/long workloads. Local **workstation = CPU-only, ≤8 cores** = pure
  logic, unit tests, small CPU-platform smoke runs. So: build + unit-test on local; run the A0 profiler,
  validate the persistent-Context / runner-inline-CV / any OpenMM change, and do the WESTPA benchmark on
  the remote. Prefer landing a refactor with live validation in the loop on the remote over building large
  unvalidated OpenMM code blind on local.
- **Branch:** `feature/hpc-cluster-enhancements` (off `main`; not pushed). Cadence: small CPU-tested
  commits; validate OpenMM/GPU paths on the host before merging → `devel` → `main`.
- **Needs GPU/host validation** (no OpenMM/mdtraj/ruptures + no GPU in this dev env): restart HDF5
  run-continuation, persistent-Context on real CUDA, incremental-projection numerical equivalence, and
  re-verifying Tables II–III after any numeric-touching change. Host tests are ready (`-m integration`).
- **Author decisions (see `docs/MANUSCRIPT_CONSISTENCY.md`):** D-1 bin factors (paper 1.5/0.75 vs code
  1.2/0.8), D-2 changepoint (paper PELT vs code `KernelCPD(pen=0.01)`), Trp-cage `D_merge` (0.6 vs supp
  0.08). All now config-selectable; confirm which produced the published numbers.

## Conventions & gotchas
- `increment`'s sign selects both target direction **and** the I0 form (`metric.scale_weights`): for Q
  `+1`=folding; for RMSD `-1`=folding (and `rmsd_backbone` requires `target`).
- Keep new numeric logic in the pure-numpy cores (no OpenMM/mdtraj/MDAnalysis/ruptures at import) with a
  `Scripts/tests/` test; heavy MD imports stay lazy. Reproducibility flows from the single seeded
  `resampler._rng` — don't reintroduce unseeded RNG.
- Output: `Systems/<name>/<output_folder>/simdata_run<run>_steps<n_steps>_cycs<n_cycles>/` →
  `pkls/` (checkpoints), `trajectories/`, `wepy.results.h5`, `Info_<run>.txt`, `dashboard.log`. A fresh
  run backs up an existing dir; `--restart` reuses it in place.
