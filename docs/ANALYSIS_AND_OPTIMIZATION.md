# CoWERA — Code Analysis, Bug Fixes & Performance Optimization

This document records a critical analysis of the CoWERA implementation in the
context of the published method (Shahid, Maity & Chakrabarty, *J. Chem. Phys.*
**164**, 134111, 2026), the bugs found, the optimizations applied, and the
staged roadmap for the remaining HPC / multi-GPU work.

The work is being landed in **phases** on the `devel` branch; each phase is a
self-contained, reviewed change set.

---

## 1. How the code maps to the paper

| Paper concept | Equation / Appendix | Code |
|---|---|---|
| Progress coordinate `P` (Q or backbone RMSD) | A1, A2 | `cowera/features.py` (`best_hummer_q`, `RMSD_Backbone`, `compute_q`) |
| Relevant history `τ_i` via changepoints | Appendix B | `cowera/metric.py::phase_calculation`, `relevant_change_points` |
| Phase `ϕ_i` (mean signed bin displacement) | Eq. 3 / C2 | `metric.phase_from_bins` |
| Normalized phase `ϕ̃_i` | Eq. C3 | `metric.scale_phases` |
| Intensity `I_i = I0_i √(1+ϕ̃_i)`, normalized | Eq. 4 / C4 | `metric.compute_intensity`, `scale_weights` (I0) |
| Total variation, accept `V_new/(V_new+V_cur)` | Eq. 5, 6 / Appendix E | `cowera/resampler.py::decide`, `_calcvariation` |
| Merge candidate proximity `D_ij ≤ D_merge` | Appendix D | `resampler.decide` (`merge_mask`), `metric.pairwise_distance_matrix` |
| Adaptive bin count (20%/80% rule) | Appendix F | `metric.intensity_calculation` |
| Warping at `D_warp`, MFPT via Hill relation | Eq. 7 | `cowera/warper.py::TargetBC` |

---

## 2. Bugs found and fixed (Phase 1)

### 2.1 Operator-precedence bug in the relevant-history window
`metric.py` computed
```python
np.all(changes[:-1] - n_d) > 0          # WRONG: np.all(...) is a bool, then > 0
```
instead of the intended
```python
np.all((changes[:-1] - n_d) > 0)
```
The first form reduces the array to a single boolean *before* the comparison,
so the pre-transition buffer (Appendix B) was applied under the wrong
condition. Fixed and isolated in `metric.relevant_change_points`, with a
regression test (`test_relevant_change_points_precedence_fix`).

### 2.2 Divide-by-zero → NaN propagation in scaling
* `phases_scaled = 2(ϕ−ϕ_min)/(ϕ_max−ϕ_min) − 1` produced `0/0 = NaN` whenever
  **all walkers had equal phase** (e.g. all stalled — a common situation early in
  a run). The NaN then flowed into `interference`/`Intensity` and corrupted
  resampling.
* The intensity normalization `interference / interference.max()` divided by
  zero when every walker's interference was zero.

Both are now guarded (`scale_phases`, `scale_weights`, `compute_intensity`) and
return well-defined neutral/uniform values. This directly supports the paper's
central claim of **reduced run-to-run variance**, since silent NaNs are a source
of irreproducibility.

### 2.3 Empty-segment phase → NaN
`np.sign(np.diff(seg)).mean()` returned NaN when the last two changepoints were
adjacent (`len(seg) < 2`). `phase_from_bins` now treats a sub-two-point segment
as fully stalled (`0.0`).

### 2.4 Broken exception fallbacks in feature functions
Both `best_hummer_q` and `RMSD_Backbone` had bare `except Exception` blocks that
recursed with `init_file` passed as the *trajectory* and returned a mismatched
tuple shape, silently masking real errors and substituting physically wrong
values. They now: (a) surface genuine errors, (b) fall back **once** to the
initial structure with the correct return shape, and (c) guard against infinite
recursion.

### 2.5 Inconsistent RNG / reproducibility
Three RNGs (`random`, `numpy.random`, the `rand` alias) were used while only
`rand` was seeded. All stochastic resampling decisions now flow through a single
seeded `numpy.random.Generator` (`resampler._rng`), making runs reproducible
from `seed` — essential for the variance-reduction study in the paper.

---

## 3. Performance optimizations (Phase 1)

All changes below preserve the algorithm's numerical results (modulo the bug
fixes) and target the **serial CPU resampling** that runs between GPU bursts and
starves the GPUs on a multi-GPU workstation (Amdahl bottleneck).

### 3.1 Native-contact caching — the biggest easy win
`best_hummer_q` recomputed the heavy-atom contact set
(`combinations(heavy, 2)` + distance filter + reference distances `r0`) **from
scratch on every call** — i.e. for every walker, in the phase calculation, in
the image projection, and again in the warper. For Trp-cage (~122 contacts from
thousands of heavy-atom pairs) this is dozens of full rebuilds per cycle.

`features.get_native_contacts(native_file)` now computes the contact definition
**once** and caches it keyed by file. The vectorized `compute_q` core is ~**150×**
faster than the equivalent Python loop (see `tests/bench_hotpaths.py`).

### 3.2 Eliminate per-pair MDAnalysis Universe construction
`metric.image_distance` previously built **two** `mda.Universe(native_file)`
objects for **every walker pair** — `2·C(N,2)` topology parses per cycle (240 for
N=16). The backbone selection and reference template are now built once and
cached (`_get_mda_template`); `pairwise_distance_matrix` reuses them and slices
the backbone atoms directly from the coordinate arrays.

### 3.3 Cache RMSD reference structures
`RMSD_Backbone` reloaded the reference and target structures (and recomputed the
target RMSD range) on every call. These are now cached by file path
(`_load_rmsd_reference`).

### 3.4 Parallelize the per-walker phase loop
The serial `for i in range(n_walkers)` phase loop now uses `joblib.Parallel`
(threading backend — mdtraj releases the GIL), which was already imported but
unused.

### 3.5 Trim redundant work in the resampler
`get_dist` no longer keeps a dead `dl`/combinations loop; it calls the
vectorized `pairwise_distance_matrix` and the intensity calculation directly.

---

## 4. Phase 2 (landed) — in-memory CV history

The runner writes each walker's trajectory to `walker_{i}.dcd` and the resampler
previously re-read **and re-projected the entire accumulated trajectory** from
disk for every walker every cycle in `phase_calculation` — an O(T) disk read,
O(T) re-projection and O(T²) changepoint cost that grows as histories lengthen.

Phase 2 introduces `cowera/cv_history.py::CVHistory`, an in-memory per-walker CV
time series that is updated **incrementally**:

- `extend` appends only the frames produced in the latest segment (so the Q/RMSD
  projection cost per cycle drops from O(T) to O(segment));
- `reindex` reorganizes histories under clone/merge by array reassignment,
  mirroring `file_resampler.update_dcd_files` semantics in memory (decoupling the
  resampling analysis from the trajectory-file bookkeeping);
- warp resets are detected via frame-count (`incremental_window`) and clear the
  stale history;
- an optional `history_window` bounds the retained history, capping the
  changepoint cost (an algorithmic improvement over unbounded growth).

The analysis is decoupled from I/O: `metric.phase_from_projection` /
`intensity_from_projections` operate purely on in-memory arrays, while the legacy
disk methods (`phase_calculation` / `intensity_calculation`) remain as
compatibility wrappers. The resampler uses the in-memory path by default
(`use_cv_history=True`; set `False` to restore the legacy path).

`CVHistory`, `incremental_window` and the reindex logic are pure-numpy and
unit-tested (`Scripts/tests/test_cv_history.py`). The remaining disk read of the
*new* segment is the final seam removed in Phase 3 (runner-supplied CVs).

> Validation note: the `CVHistory` building blocks are unit-tested here, but the
> integrated resampler path requires the full MD stack + GPU and must be
> validated on the simulation host before this phase is merged to `main`.

## 5. Roadmap — remaining HPC / multi-GPU work (next phases)

### Phase 3 — Persistent per-GPU workers + cached OpenMM Context
The current `TaskMapper` forks a fresh process **and builds a new OpenMM
`Simulation`/CUDA Context for every walker every cycle**, recompiling kernels and
re-uploading the system each segment — the single largest source of multi-GPU
inefficiency for short resampling intervals. Switch to the persistent
`WorkerMapper` + `OpenMMGPUWorker` (already present in `wepy`) and cache the
`Context` per worker so each segment only does `setState → step → getState`.
Also trim `GET_STATE_KWARG_DEFAULTS` to fetch only positions + box vectors
(reporters save nothing else), cutting GPU→CPU transfer.

### Phase 4 — Algorithmic accelerations
* **GPU-inline CV reporting** via OpenMM `CustomCVForce` so Q/RMSD is produced
  *during* propagation, eliminating the CPU recompute + disk path entirely.
* **Online/incremental changepoint detection** to replace per-cycle
  `ruptures.KernelCPD` (≈O(T²)); only the most recent changepoint matters.
* **Lazy distance matrix**: only the low-intensity merge candidate's distances
  are needed for `merge_mask`; compute RMSD on demand instead of full O(N²).
* **Overlap** cycle-`k` CPU resampling with cycle-`k+1` GPU propagation.

---

## 6. Running the tests & benchmarks

The Phase-1 algorithm cores are pure NumPy and run without a GPU or the MD stack:

```bash
python -m pytest Scripts/tests/ -v
python Scripts/tests/bench_hotpaths.py
```

See `docs/REPRODUCING_THE_PAPER.md` for end-to-end reproduction of the chignolin
and Trp-cage kinetics on GPU hardware.
