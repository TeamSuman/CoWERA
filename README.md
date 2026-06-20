# 🧬 ComWERA
## Committor-based Weighted Ensemble Resampling Algorithm

**ComWERA** extends [**CoWERA**](https://github.com/TeamSuman/CoWERA) (*Coherence-based
Weighted Ensemble Resampling Algorithm*, J. Chem. Phys. **164**, 134111, 2026) by replacing
its sign-based temporal-coherence phase with a learned **committor** `q(x) = P(reach B before A | x)`
— the provably optimal reaction coordinate — as the live clone/merge criterion.

It is a binless, GPU-accelerated weighted-ensemble method for rare-event kinetics (protein
folding/unfolding, conformational change, ligand (un)binding) built on OpenMM and Wepy. The
committor is trained **on-the-fly from the WE trajectory pool itself**, with no reactive
trajectories required, via a cold-start bootstrap protocol.

> **Status:** active development. The published `cowera/` method runs unchanged and remains the
> default; the committor extension lives in the opt-in `Scripts/cowera_committor/` subpackage.
> Milestones M1–M3 are implemented and CPU-tested; see the roadmap below.

---

## Why committor-guided resampling?

CoWERA's coherence phase is a powerful proxy, but it is a 1D projection and inherits CV
degeneracy. The committor is the unique coordinate that, used for resampling, eliminates that
degeneracy by construction:

- **Δq replaces the sign-based phase** — a strict, continuous generalization that plugs directly
  into CoWERA's existing intensity machinery (no change to the resampler core).
- **Committor-augmented merging** — `α·D_RMSD + (1−α)·|q_i − q_j|` prevents merging kinetically
  distinct walkers that happen to share an RMSD.
- **Cold-start bootstrap** — a phase manager trains `q` during the run (structural seed → semigroup
  → shooting/AIMMD → mature), controlling how much the resampler trusts `q` at each stage.

See [`docs/COMMITTOR_EXTENSION.md`](docs/COMMITTOR_EXTENSION.md) and the validated tutorial
notebook [`Analysis/CoWERA_Committor_Tutorial.ipynb`](Analysis/CoWERA_Committor_Tutorial.ipynb).

---

## Repository layout

```
ComWERA/
├── Scripts/
│   ├── cowera/                 # core CoWERA method (resampler, metric, features, warper, cv_history)
│   ├── cowera_committor/       # committor extension (opt-in; numpy-only near-term)
│   │   ├── committor_model.py      # MLP committor (GVP-GNN backend planned)
│   │   ├── committor_metric.py     # CommittorDistances: q-inference + Δq + committor merge
│   │   ├── committor_resampler.py  # CommittorResampler(CoWERAResampler)
│   │   ├── phase_manager.py        # cold-start BootstrapPhaseManager
│   │   ├── bootstrap.py            # Phase-0 structural interpolants
│   │   ├── committor_train.py      # buffers → training; retrain orchestration
│   │   ├── featurizer.py, committor_data.py, toy_systems.py
│   ├── wepy/                   # vendored Wepy engine
│   ├── tests/                  # CPU unit tests + benchmarks (no GPU/MD required)
│   ├── run_cowera.py               # standard CoWERA entry point
│   ├── run_cowera_committor.py     # committor-guided entry point
│   ├── config_template.yml, config_committor.yml
├── Systems/                    # example systems (chignolin, Trp-cage)
├── Analysis/                   # analysis + tutorial notebooks
├── docs/                       # analysis, reproduction, committor, roadmap
└── env/environment.yml
```

---

## Installation

```bash
git clone https://github.com/TeamSuman/ComWERA.git
cd ComWERA
conda env create -f env/environment.yml -n comwera
conda activate comwera
```

The committor core (`cowera_committor`) needs only `numpy`/`scipy` near-term; the planned
GVP-GNN backend will add `torch`/`torch-geometric`.

---

## Quick start

**Standard CoWERA run** (coherence-guided):
```bash
python ./Scripts/run_cowera.py --config ./Systems/<system>/config.yml
```

**Committor-guided run** (opt-in):
```bash
cp Scripts/config_committor.yml Systems/<system>/config_committor.yml
# edit it for your system, then:
python ./Scripts/run_cowera_committor.py --config ./Systems/<system>/config_committor.yml
```

GPU execution with CUDA MPS is recommended for packing multiple walkers per device:
```bash
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20
```

See [`docs/REPRODUCING_THE_PAPER.md`](docs/REPRODUCING_THE_PAPER.md) for chignolin / Trp-cage
configs that reproduce the published kinetics (Tables II–III).

---

## Testing

The algorithm cores are pure NumPy and run without a GPU or the MD stack:

```bash
python -m pytest Scripts/tests/ -v        # 87 tests
python Scripts/tests/bench_hotpaths.py    # hot-path benchmarks
```

These cover the CoWERA bug-fix edge cases, the in-memory CV history, the committor model
(finite-difference gradient check + analytic double-well recovery, MAE ≈ 0.014), the Δq resampling
path, and the cold-start phase manager.

---

## Documentation

- [`docs/COMMITTOR_EXTENSION.md`](docs/COMMITTOR_EXTENSION.md) — committor extension design, status, figures.
- [`docs/CoWERA_Implementation_Plan.md`](docs/CoWERA_Implementation_Plan.md) — full status (done / future).
- [`docs/DEVELOPMENT_ROADMAP.md`](docs/DEVELOPMENT_ROADMAP.md) — detailed plans for the next milestones.
- [`docs/ANALYSIS_AND_OPTIMIZATION.md`](docs/ANALYSIS_AND_OPTIMIZATION.md) — performance refactor & bug fixes.
- [`docs/REPRODUCING_THE_PAPER.md`](docs/REPRODUCING_THE_PAPER.md) — reproduce the benchmarks.

---

## Status & roadmap

| Milestone | Status |
|---|---|
| Performance: bug fixes + hot-path optimization | ✅ done |
| Performance: in-memory CV history | ✅ done |
| Committor M1: core model + validation | ✅ done |
| Committor M2: Δq live resampling path | ✅ done |
| Committor M3: cold-start bootstrap + on-the-fly training | ✅ done |
| Performance Phase 3: persistent per-GPU contexts + runner CV hook | ⬜ next |
| Committor M4: shooting-point (AIMMD) augmentation | ⬜ next |
| GVP-GNN backend; FES/interpretability; ligand binding; multi-state | ⬜ future |

Details and execution plans for the next milestones are in
[`docs/DEVELOPMENT_ROADMAP.md`](docs/DEVELOPMENT_ROADMAP.md).

---

## Citation

If you use ComWERA, please cite the CoWERA paper:

> S. Shahid, D. Maity, S. Chakrabarty, *CoWERA: A temporal coherence guided binless resampling
> algorithm for weighted-ensemble based estimation of rare-event kinetics*, J. Chem. Phys. **164**,
> 134111 (2026). https://doi.org/10.1063/5.0320586

---

## Acknowledgements

Built on [Wepy](https://github.com/ADicksonLab/wepy) and [OpenMM](https://openmm.org/).
Developed in the group of Suman Chakrabarty, S. N. Bose National Centre for Basic Sciences, Kolkata.
