# Manuscript ↔ Code Consistency

A cross-check of the CoWERA implementation against the manuscript
(Shahid, Maity & Chakrabarty, *J. Chem. Phys.* **164**, 134111, 2026) and its
supplement. Each row is a place where the paper text, the supplement, the shipped
configs/docs, and the code disagreed. Code and repository docs have been
reconciled where safe; **items marked "author decision" need the authors to
confirm which value produced the published numbers before the manuscript text is
corrected.**

The code paths for the parameters below are now all configurable, so either the
paper value or the code default can be selected from `config.yml`.

## Resolved in the repository

| Topic | Was | Now | Where |
|---|---|---|---|
| Chignolin temperature | `docs/REPRODUCING_THE_PAPER.md` said 340 K | 275 K (matches `supp.tex` NVT 275 K and `Systems/chignolin/system.py`) | `docs/REPRODUCING_THE_PAPER.md` |
| Conda env in job script | PBS script activated `wepy` | `cowera` (matches `env/environment.yml`) | `Scripts/sample_job_multirun.sh` |
| Run invocation | PBS script ran `run_cowera.py` from `$PBS_O_WORKDIR` | runs `./Scripts/run_cowera.py` from repo root (matches README) | `Scripts/sample_job_multirun.sh` |
| Adaptive-bin factors dropped | in-memory path ignored the factors | forwarded + exposed as `bin_increase_factor`/`bin_decrease_factor` | `Scripts/cowera/metric.py`, `resampler.py`, `config_template.yml` |

## Author decision needed (manuscript vs code)

| Topic | Manuscript | Code | Config knob to reconcile |
|---|---|---|---|
| **D-1: adaptive-bin factors** | Appendix F: γ↑ = **1.5**, γ↓ = **0.75** | defaults **1.2 / 0.8** | set `bin_increase_factor: 1.5`, `bin_decrease_factor: 0.75` to match the paper |
| **D-2: changepoint algorithm** | Appendix B: **PELT** (linear kernel) | `ruptures.KernelCPD(kernel="linear", min_size=2).predict(pen=0.01)` — a different search, with an undocumented penalty `pen=0.01` | — (no knob yet; confirm which was used for the published runs) |
| **Trp-cage `D_merge`** | main-text Table I: **0.6** | — | supplement says **0.08**; `config_template.yml` said **0.3**; `REPRODUCING_THE_PAPER.md` uses **0.6**. 0.6 nm pairwise-RMSD is the internally consistent value; confirm and correct the supplement |
| Merge-partner selection | Appendix D: merge with a *nearby* walker within `D_merge` | a **uniformly random** eligible walker within `D_merge` (`self._rng.choice`) | — (confirm intent: random-within-cutoff vs closest) |
| Initial intensity `I0` | Appendix C: "uniformly one **or** a function of the projection" | only the projection form was implemented | exposed as `i0_mode` (see Phase 3.3 / `config_template.yml`) |

## Notes
- The `pen=0.01` changepoint penalty and the random (vs closest) merge-partner
  choice are undocumented in the manuscript; if they are intended, add them to the
  methods/appendix; if not, adjust the code.
- Once D-1/D-2 are settled, update Appendix F / Appendix B and this table together.
