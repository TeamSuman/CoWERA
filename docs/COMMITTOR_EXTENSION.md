# CoWERA-Committor — Learned Committor as the Resampling Criterion

This document describes the **opt-in** committor extension of CoWERA and the
status of its first milestone (M1). The goal is to replace/augment CoWERA's
sign-based temporal-coherence phase `φ` with a learned **committor**
`q(x) = P(reach B before A | x)` — the provably optimal reaction coordinate
(backward Kolmogorov PDE) — used as the clone/merge criterion.

The full design is captured in the planning documents; this file tracks what is
implemented and how to validate it. The published `cowera/` path is unchanged
and remains the default; the committor lives in a separate subpackage
`Scripts/cowera_committor/` and pulls in no machine-learning dependency unless
explicitly used.

## Why this fits CoWERA cleanly

Mechanically, the committor is just a different progress coordinate
`P(x) = q(x)` feeding the **same** phase→intensity machinery, with the continuous
committor displacement `Δq` replacing the sign-based phase
(`np.sign(np.diff(...))`, which `Δq` strictly generalizes). The performance
refactor (PR #2) already created the seams this reuses:

| Need | Reused seam |
|---|---|
| Per-frame CV series + clone/merge reindex + warp reset | `cowera/cv_history.py::CVHistory` (store `q(x)` per frame) |
| `Δq` → intensity (Eqs. 3–6) | `cowera/metric.py`: `intensity_from_projections`, `scale_phases`, `compute_intensity` |
| Per-frame CV production seam (→ model inference) | `cowera/metric.py::project_new_frames` |
| Clone/merge optimizer + merge proximity | `cowera/resampler.py::decide` (augment merge distance with `|q_i−q_j|`) |
| Reactive (B-touch) detection | `cowera/warper.py::TargetBC._progress` |

Because of this, swapping `φ → Δq` requires **no change** to `decide()`; the
committor path will subclass `Calculate_Distances` / `CoWERAResampler`.

## Milestone M1 (implemented) — committor core

`Scripts/cowera_committor/`:

- **`featurizer.py`** — SE(3)-invariant features. `DistanceFeaturizer` uses
  internal pairwise distances (rotation/translation invariant; reuse
  `cowera.features.get_native_contacts` for the pairs); `IdentityFeaturizer`
  passes toy-system CV coordinates through.
- **`committor_model.py`** — `CommittorModel` interface + `MLPCommittor`, a
  dependency-free MLP with **manual backprop** and Adam. Trained with the
  WE-weighted variational (Dirichlet / semigroup) loss plus boundary and
  optional AIMMD shooting terms:

  ```
  L = w_b [ λ_A <w q²>_A + λ_B <w (1−q)²>_B ]            (boundaries)
    + w_s <w (q(x_t) − q(x_{t+τ}))²>                      (semigroup)
    + w_a (−<w [n_A log(1−q) + n_B log q]>)               (AIMMD)
  ```

  Every term carries the WE walker weight `w` (inverse-probability weighting),
  the subtle correctness point from the cold-start protocol. The output head is
  shaped so a softmax multi-state head (`n_states > 2`) drops in later.

- **`committor_data.py`** — `BoundaryBuffer` (permanent), `SemigroupBuffer` and
  `ShootingBuffer` (sliding window), all WE-weighted.

- **`toy_systems.py`** — analytic validation: symmetric double well
  `V(x)=(x²−1)²`, overdamped Langevin trajectory generator (source of semigroup
  pairs), and the exact 1D committor by quadrature
  `q(x)=∫_a^x e^{βV}/∫_a^b e^{βV}`.

### M1 validation (CPU, no GPU/MD)

`Scripts/tests/test_committor_*.py` (38 tests in the suite total):

- **Finite-difference gradient check** — analytic gradients match numerical
  gradients for all loss terms (guarantees the manual backprop is correct).
- **Double-well committor recovery** — trained on semigroup + boundary data, the
  model reproduces the analytic committor with **MAE ≈ 0.014** (target < 0.02),
  `q(0) ≈ 0.48` (barrier ≈ 0.5), correlation 0.999.
- Featurizer SE(3) invariance; buffer sliding-window / weighting; toy-system
  ground truth (boundary values, symmetry, monotonicity).

Run:
```bash
python -m pytest Scripts/tests/ -v
```

### Tutorial notebook & validation figures

A full, executed walkthrough is in
[`Analysis/CoWERA_Committor_Tutorial.ipynb`](../Analysis/CoWERA_Committor_Tutorial.ipynb)
(needs only `numpy` + `matplotlib`): it builds the benchmark, generates WE-style
training data, trains the committor, validates it, and shows the Δq → intensity
mapping. Generated figures (`docs/figures/`):

**Learned vs. exact committor** — MAE ≈ 0.014:

![Committor recovery](figures/committor_validation.png)

**Committor histogram test** — empirical $p_B$ from transition-state candidates
clusters at 0.5 (mean ≈ 0.53, std ≈ 0.07), the standard dynamical validation:

![Histogram test](figures/committor_histogram_test.png)

**Δq → CoWERA intensity** — committor displacement feeds the production
`scale_phases` / `compute_intensity` path; the brightest (most forward-progressing)
walkers are cloned:

![Intensity](figures/committor_intensity.png)

Other figures: `committor_doublewell_system.png`, `committor_training_data.png`,
`committor_loss.png`.

## Cold-start protocol (designed; staged for M3+)

`BootstrapPhaseManager` will drive the four-phase bootstrap that needs **no
reactive trajectories**:

- **Phase 0** structural interpolants (no MD) → coarse `q⁰`;
- **Phase 1** semigroup training on WE segment pairs (conservative merge);
- **Phase 2** shooting-point AIMMD from first reactive fragments (histogram test);
- **Phase 3** mature self-consistent training (sliding window, JSD convergence).

## Statistical validity

On-the-fly committor retraining does **not** bias WE: statistical exactness comes
solely from weight conservation during clone/merge. The resampling criterion
affects **efficiency, not correctness**, so changing it mid-run (as CoWERA
already does for adaptive binning) is valid.

## Roadmap (next milestones)

- **M2** — `committor_metric.py`: `Δq` intensity path reusing
  `intensity_from_projections`; `CVHistory` keyed on `q`.
- **M3** — `committor_resampler.py` + `phase_manager.py` (Phase 0/1) +
  committor-augmented merge distance; opt-in via `config_committor.yml`.
- **M4** — reactive-fragment harvest + shooting interfaces (depends on the
  runner CV/burst hook from performance Phase 3).
- **Stage 2+** — GVP-GNN backend (torch-geometric) behind `CommittorModel`;
  live async retraining; chignolin/Trp-cage benchmarks; FES along `q`;
  multi-state vector/local committor for multi-basin landscapes.
