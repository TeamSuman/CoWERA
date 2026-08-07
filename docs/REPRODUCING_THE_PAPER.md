# Reproducing the CoWERA Paper Results

This guide reproduces the chignolin and Trp-cage folding/unfolding kinetics
reported in Shahid, Maity & Chakrabarty, *J. Chem. Phys.* **164**, 134111 (2026)
(Tables I–III, Figs. 3–5), using the parameters from **Table I** of the paper.

> Requires a CUDA GPU and the full environment (`env/environment.yml`:
> OpenMM, mdtraj, MDAnalysis, ruptures). The algorithm-core unit tests in
> `Scripts/tests/` run without any of these — see the bottom of this page.

---

## 1. Parameter mapping (Table I → `config.yml`)

CoWERA's config fields relate to the paper's symbols as follows (integration
timestep `dt = 0.002 ps`):

| Paper symbol | Meaning | config field | Conversion |
|---|---|---|---|
| `ΔT` | resampling interval (ps) | `n_steps` | `n_steps = ΔT / dt` |
| `f` | CV save interval (ps) | `save_freq` | `save_freq = f / dt` (gives `n_d = n_steps/save_freq`) |
| `P` | progress coordinate | `sel_feat` | `Q → best_hummer_q`, `RMSD → rmsd_backbone` |
| `D_warp` | warping cutoff from target | `d_warped` | direct |
| `D_merge` | merge cutoff | `d_merge` | direct |
| `p_max` | max walker probability | `pmax` | direct |
| `N_c` | # native contacts → bins | `n_bins`, `max_bins` | 29 (chignolin), 122 (Trp-cage) |
| direction | folding vs unfolding | `increment` | see below |

**Direction / `increment` and `I0`.** The sign of `increment` selects the
target direction *and* the functional form of the initial intensity `I0`
(implemented in `metric.scale_weights`):

* `increment = +1` → `I0 = (P − P_min)/(P_max − P_min)`
* `increment = -1` → `I0 = 1 − (P − P_min)/(P_max − P_min)`

Matching Table I:

| System | Process | `sel_feat` | `increment` |
|---|---|---|---|
| Chignolin | Folding | `rmsd_backbone` | `-1` |
| Chignolin | Unfolding | `rmsd_backbone` | `+1` |
| Trp-cage | Folding | `best_hummer_q` | `+1` |
| Trp-cage | Unfolding | `best_hummer_q` | `-1` |

All four use **16 walkers** and `mode: probabilistic`.

---

## 2. Example configs

Place these as `Systems/<system>/config.yml`. Files (`*.gro`, `topol.top`,
`system.py`, etc.) must already be present per the main `README.md`.

### Trp-cage folding (Table I, row 3; Table III: MFPT ≈ 14 µs)
```yaml
system: "Trp-cage"
dir: "./Systems/TC10b Trp-cage"

num_walkers: 16
pmax: 0.25
run: "fold_0"
n_steps: 50000      # ΔT = 100 ps
n_cycles: 20000
save_freq: 5000     # f = 10 ps  -> n_d = 10

start: "unfolded.gro"
topol: "topol.top"
target: "folded.gro"
native: "folded.gro"

sel_feat: "best_hummer_q"
mode: "probabilistic"
distance_criterion: "pairwise_rmsd"

d_merge: 0.6
d_warped: 0.3

temp: 290.0
gpu_ids: [0]

n_bins: 122
max_bins: 122
increment: 1
output_folder: "folding_runs"
```

### Trp-cage unfolding (Table I, row 4; Table III: MFPT ≈ 3 µs)
Same as above with:
```yaml
run: "unfold_0"
n_steps: 25000      # ΔT = 50 ps
start: "folded.gro"
target: "unfolded.gro"
increment: -1
output_folder: "unfolding_runs"
```

### Chignolin folding (Table I, row 1; Table II: K ≈ 0.96×10⁷ s⁻¹)
The runnable version ships as `Systems/chignolin/config.yml` (uses the shipped
AMBER inputs `chignolin_*.pdb` + `chignolin.prmtop`). Table-I parameters:
```yaml
system: "chignolin"
dir: "./Systems/chignolin"

num_walkers: 16
pmax: 0.20
run: "fold_0"
n_steps: 1000       # ΔT = 2 ps
n_cycles: 20000
save_freq: 100      # f = 0.2 ps -> n_d = 10

start: "chignolin_unfolded.pdb"
topol: "chignolin.prmtop"
target: "chignolin_folded.pdb"
native: "chignolin_folded.pdb"

sel_feat: "rmsd_backbone"
mode: "probabilistic"
distance_criterion: "euclidean"   # |ΔRMSD| in CV space (shipped config); "pairwise_rmsd" also valid

d_merge: 0.1        # nm
d_warped: 0.05      # nm
temp: 275.0         # matches supp.tex (NVT, 275 K) and Systems/chignolin/system.py
gpu_ids: [0]

n_bins: 29
max_bins: 29
increment: -1
output_folder: "folding_runs"
```

### Chignolin unfolding (Table I, row 2; Table II: K ≈ 0.69×10⁷ s⁻¹)
Same as chignolin folding with:
```yaml
run: "unfold_0"
n_steps: 5000       # ΔT = 10 ps
save_freq: 1000     # f = 2 ps -> n_d = 5
start: "chignolin_folded.pdb"
target: "chignolin_unfolded.pdb"
d_warped: 0.145     # nm
increment: 1
output_folder: "unfolding_runs"
```

---

## 3. Running

Enable CUDA MPS to pack multiple walkers onto each physical GPU, then launch:

```bash
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20   # tune to GPU memory / walker count

python ./Scripts/run_cowera.py --config ./Systems/TC10b\ Trp-cage/config.yml
```

For statistics matching the paper, run **multiple independent replicas** by
changing `run:` (e.g. `fold_0 … fold_4`) — the paper reports block-averaged
estimates with confidence intervals across replicas.

On a multi-GPU workstation, list every device in `gpu_ids` (e.g. `[0, 1, 2, 3]`);
walkers are distributed across the listed GPUs.

### 3.1 Running on HPC clusters (SLURM / PBS)

Ready-to-edit batch scripts are provided:

```bash
# SLURM — one run, or 5 replicas as a job array (distinct run id per task):
sbatch --export=ALL,CONFIG=./Systems/chignolin/config.yml Scripts/slurm/cowera.sbatch
sbatch --array=0-4 --export=ALL,CONFIG=./Systems/chignolin/config.yml Scripts/slurm/cowera.sbatch

# PBS:
qsub -v CONFIG=./Systems/chignolin/config.yml,REPO_ROOT=$PWD Scripts/sample_job_multirun.sh
```

Both scripts activate the `cowera` env, run from the repository root, start **and
tear down** CUDA MPS, and keep BLAS/OpenMP threads inside the allocation. Notes:

- **Device ids.** `gpu_ids` must index the **scheduler-visible** devices — SLURM/PBS
  renumber `CUDA_VISIBLE_DEVICES` to `0..N-1`, so use those, not physical ids.
  Repeat ids (`[0,0,0]`) only to pack walkers on one GPU via MPS.
- **CPU-only nodes.** Set `platform: "CPU"`, `gpu_ids: []`, and `n_workers: <cores>`.
- **Wall-time / restart.** Set a realistic `--time`; if a job is cut off, resume with
  the same config plus `--restart` (or `restart: true`) — it continues from the newest
  checkpoint (written every `checkpoint_freq` cycles) without clobbering prior output.
- **Shared filesystems (Lustre/GPFS).** CoWERA writes many small per-walker DCD files
  and appends to them every cycle, which stresses the metadata server. Set
  `scratch_dir: "$TMPDIR"` in the config to stage that trajectory I/O on node-local
  disk; it is synced back to the output directory on exit. Checkpoints and the
  results file stay on the (durable) output directory so restart still works.
- **CPU oversubscription.** The per-cycle analysis honors `SLURM_CPUS_PER_TASK` /
  `PBS_NP` for its joblib pool, so keep `--cpus-per-task` accurate.

---

## 4. Analysis (Tables II–III, Figs. 3–5)

After a run, results live in
`Systems/<system>/<output_folder>/simdata_run<run>_steps<n_steps>_cycs<n_cycles>/`
(`wepy.results.h5`, `trajectories/`, `Info_<run>.txt`).

Use `Analysis/WE_analysis.ipynb` to:

1. Accumulate warped-walker weight `Σ_t w_target(t)` and apply the Hill relation
   `MFPT = T_total / Σ_t w_target(t)` (Eq. 7); `k = 1/MFPT`.
2. Interpolate aggregated probability onto a common time grid (20 ps window for
   chignolin, 0.5 ns for Trp-cage) across replicas.
3. Identify the converged regime (chignolin: ~6–7.5 ns; Trp-cage: ~23–74 ns) and
   block-average beyond it for the rate/MFPT and confidence intervals.

> **Use the total simulated time in the denominator.** `T_total` in Eq. (7) is the
> aggregate time actually simulated (`n_cycles × ΔT × num_walkers`), **not** the time
> of the last recycling event. Time after the final event produced no flux but was
> still simulated; omitting it inflates the rate (up to ~1.6× on runs whose last
> event lands early). The notebook reports both conventions so the difference is
> visible.

> **Report the effective sample size with every rate.** The flux is a weighted sum
> of rare events, so a run can accumulate many recycling events and still rest on a
> handful of high-weight walkers. `Scripts/tools/mfpt_estimators.py` prints the Kish
> ESS alongside the rate; treat an estimate with an ESS of only a few events as
> preliminary regardless of how many events were recorded.

### Run lengths and replicas

The published runs used the following cycle counts (supplement, Table S1):

| System / process | ΔT | cycles per run | replicas |
|---|---|---|---|
| Chignolin folding | 2 ps | 10 254 – 13 042 | 5 |
| Chignolin unfolding | 10 ps | 2 227 – 2 274 | 5 |
| Trp-cage folding | 100 ps | 997 – 1 053 | 5 |
| Trp-cage unfolding | 50 ps | 621 – 626 | 5 |

Chignolin folding in particular needs **≥10 000 cycles**; shorter runs are not
converged. Always run **several independent replicas** — set a different `seed` and
`run` label for each (`Scripts/tools/make_replica_config.py` generates them) — and
block-average with `Scripts/tools/replica_stats.py`. A single run does not carry a
meaningful confidence interval.

### Settings that affect reproduction

The defaults reproduce the published behaviour; these are the knobs that matter if
you change them:

| Option | Default | Note |
|---|---|---|
| `relevant_history_mode` | `published` | Reproduces the released phase-window guard. `strict` shortens the phase window and changes the resampling — do not use for reproduction. |
| `merge_partner` | `random` | The published runs merged into a random eligible walker. `closest` samples more efficiently but can stall the ensemble on some seeds. |
| `bin_increase_factor` / `bin_decrease_factor` | `1.2` / `0.8` | The values used by the released code. |
| `changepoint_algo` | `kernelcpd` | The search used by the released code; `pelt` is far slower. |
| `changepoint_window` | `1000` | Bounds the changepoint cost; expands adaptively, and was verified to give the same rate as the unbounded scan. |

**Expected targets**

| System | Process | Reference (unbiased MD) | CoWERA |
|---|---|---|---|
| Chignolin | Unfolding | K = 1.3 (0.9, 1.8) ×10⁷ s⁻¹ | 0.69 (0.56, 0.83) |
| Chignolin | Folding | K = 0.71 (0.44, 1.24) ×10⁷ s⁻¹ | 0.96 (0.74, 1.18) |
| Trp-cage | Folding | MFPT = 14 ± 4 µs | 14.38 ± 0.27 |
| Trp-cage | Unfolding | MFPT = 3 ± 1 µs | 3.50 ± 0.11 |

---

## 5. Validating the algorithm cores without a GPU

The coherence/intensity machinery (Eqs. 3–6, Appendices B–F) is unit-tested in
pure NumPy and runs anywhere:

```bash
python -m pytest Scripts/tests/ -v
python Scripts/tests/bench_hotpaths.py
```

These pin down the phase/intensity formulas and the edge-case bug fixes
(degenerate phases, empty segments, changepoint window), and demonstrate the
CPU hot-path speedups described in `docs/ANALYSIS_AND_OPTIMIZATION.md`.
