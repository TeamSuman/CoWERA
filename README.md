# 🧬 CoWERA  
## Coherence-based Weighted Ensemble Resampling Algorithm

CoWERA is a **binless weighted-ensemble (WE) resampling algorithm** for efficient estimation of rare-event kinetics (e.g., protein folding/unfolding) using OpenMM and Wepy.

It is designed for GPU-accelerated molecular simulations and rare-event rate calculations.

---

# 🚀 Complete Workflow

This guide walks through:

1. Cloning the repository  
2. Creating the environment  
3. Preparing a system  
4. Configuring the simulation  
5. Running CoWERA  
6. GPU setup (CUDA + MPS)  
7. Analysing results  

---

# 1️⃣ Clone the Repository

```bash
git clone https://github.com/Shaheerah3007/CoWERA.git
cd CoWERA
```

---

# 2️⃣ Create the Conda Environment

An environment file is provided in:

```
env/environment.yml
```

Create the environment:

```bash
conda env create -f env/environment.yml -n cowera
```

Activate it:

```bash
conda activate cowera
```

---

# 3️⃣ Repository Structure

```
CoWERA/
│
├── Scripts/                  # CoWERA + Wepy workflow scripts
│   ├── run_cowera.py         # Main execution script
│   ├── config_template.yml   # Fully documented configuration template
│   ├── cowera/               # The CoWERA method (resampler, metric, features)
│   ├── wepy/                 # Vendored Wepy weighted-ensemble engine
│   ├── tools/                # Analysis + profiling utilities (see section 11)
│   ├── tests/                # Unit tests (pure NumPy, no GPU/MD needed)
│   ├── slurm/                # SLURM batch example
│   └── sample_job_multirun.sh   # PBS batch example
│
├── docs/
│   └── REPRODUCING_THE_PAPER.md    # Reproduce the published kinetics
│
├── env/
│   └── environment.yml
│
├── Analysis/                 # Post-processing notebooks
│
└── Systems/
    └── system_name/
        ├── system.py
        ├── config.yml
        ├── folded.gro
        ├── unfolded.gro
        ├── topol.top
        ├── *.itp
        └── forcefield files (if required)
```

---

# 4️⃣ Prepare a System

Inside `Systems/`, create a folder for your system:

```
Systems/system_name/
```

## Required Files

- `system.py`
- `config.yml`
- `folded.gro`
- `unfolded.gro`
- `topol.top`
- Any `.itp` or forcefield files referenced in the topology

---

## 🧪 system.py Requirement

Each system must define:

```python
def make_system(top, temp):
    """
    Parameters:
        top  : OpenMM topology object
        temp : temperature in Kelvin

    Returns:
        system, integrator
    """
    ...
    return system, integrator
```

This function must construct:

- The OpenMM `System`
- The `Integrator` (e.g., Langevin, NPT, implicit solvent, etc.)

---

# 5️⃣ Create the Configuration File

A fully commented template is provided:

```
Scripts/config_template.yml
```

Copy it:

```bash
cp Scripts/config_template.yml Systems/system_name/config.yml
```

Edit it according to your system.

---

## Example `config.yml`

```yaml
system: "Trp-cage"
dir: "./Systems/TC10b Trp-cage"

num_walkers: 16
pmax: 0.20
run: "test_0"
n_steps: 50000
n_cycles: 10000

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
gpu_ids: [0] # if CUDA MPS enabled add 0 based on percentage

save_freq: 5000
n_bins: 122
max_bins: 122

increment: 1
output_folder: "folding_runs"
```

---

## 🔑 Important Parameters

| Parameter | Description |
|------------|-------------|
| `num_walkers` | Number of WE trajectories |
| `n_steps` | MD steps between resampling |
| `n_cycles` | Total WE cycles |
| `sel_feat` | `"best_hummer_q"` or `"rmsd_backbone"` |
| `mode` | `"greedy"` or `"probabilistic"` |
| `distance_criterion` | `"pairwise_rmsd"` or `"euclidean"` |
| `d_merge` | Merge cutoff distance |
| `d_warped` | Warping boundary cutoff |
| `increment` | +1 or −1 (direction of progress) |
| `gpu_ids` | CUDA device IDs |

---

# 6️⃣ Run the Simulation

From the repository root:

```bash
python ./Scripts/run_cowera.py --config ./Systems/system_name/config.yml
```

Replace `system_name` accordingly.

---

# 7️⃣ GPU Usage (Highly Recommended)

CoWERA is designed for GPU execution.

To efficiently run multiple walkers on a single GPU, enable CUDA MPS:

```bash
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20
```

Adjust the percentage depending on:

- GPU memory  
- Number of walkers  
- System size  

Then run normally:

```bash
python ./Scripts/run_cowera.py --config ./Systems/system_name/config.yml
```

---

# 8️⃣ Output Structure

Results are stored in:

```
Systems/system_name/output_folder/
    simdata_runtest_{run}_steps{n_steps}_cycs{n_cycles}/
```

This directory contains:

```
pkls/
trajectories/
dashboard.log
Info_test_{run}.txt
wepy.dash.org
wepy.results.h5
```

---

# 9️⃣ Trajectory Files

The `trajectories/` folder contains:

- Current walker trajectories  
- Productive trajectories  

These are saved in **DCD format** and can be analysed using:

- MDAnalysis  
- MDTraj  
- Any compatible molecular analysis toolkit  

---

# 🔬 1️⃣0️⃣ Analysis

All analysis notebooks are provided in:

```
Analysis/
```

You can analyse:

- MFPT (mean first-passage time)  
- Rate constants  
- Convergence behaviour  
- Warping statistics  

The main results file is:

```
wepy.results.h5
```

This file contains:

- Warping events  
- Walker weights  
- Resampling history  
- Full WE trajectory data  

---

---

# 🧪 1️⃣1️⃣ Tests and Analysis Utilities

## Running the tests

The algorithm cores are unit-tested in pure NumPy and need **no GPU and no MD stack**:

```bash
python -m pytest Scripts/tests/ -q
```

Tests that require the full MD stack (OpenMM / mdtraj) are marked `integration` and
are deselected by default. On a host with the environment installed:

```bash
python -m pytest Scripts/tests/ -m integration
```

## Analysis utilities (`Scripts/tools/`)

| Tool | Purpose |
|---|---|
| `mfpt_running.py` | Running MFPT / rate vs simulated time from `wepy.results.h5` (Hill relation) |
| `mfpt_estimators.py` | Compare rate estimators and report the **effective sample size (ESS)** |
| `replica_stats.py` | Block-average independent replicas → mean rate + 95% confidence interval |
| `make_replica_config.py` | Generate per-replica configs with distinct seed + run label |
| `bench_run.py` | Per-cycle wall-clock breakdown (set `profile: true` in the config) |
| `gpu_monitor.py` | Background GPU utilisation sampler |

Example — running rate curve and a replica-averaged estimate:

```bash
python Scripts/tools/mfpt_running.py --h5 <output>/wepy.results.h5 --n-walkers 16 --n-steps 1000
python Scripts/tools/replica_stats.py --n-walkers 16 --n-steps 1000 --h5 run*/wepy.results.h5
```

## 📏 Reporting a rate correctly

Two points matter more than they might appear:

1. **Divide by the total simulated time**, not by the time of the last recycling
   event. Simulated time after the final event produced no flux but was still paid
   for; excluding it inflates the rate.
2. **Quote the ESS alongside the rate.** The flux is a *weighted* sum of rare
   events, so a run with many recycling events can still have a small effective
   sample size if a few high-weight walkers dominate. `mfpt_estimators.py` reports
   both, and a rate carrying an ESS of only a few events should be treated as
   preliminary.

For statistically meaningful kinetics, run **several independent replicas**
(different `seed` and `run` label) and block-average them with `replica_stats.py`
rather than quoting a single run.

---

# 📖 Reproducing the published results

See [`docs/REPRODUCING_THE_PAPER.md`](docs/REPRODUCING_THE_PAPER.md) for the
parameter mapping (paper Table I → `config.yml`), ready-to-use configurations for
chignolin and Trp-cage folding/unfolding, and the analysis procedure.

---

# 📌 Summary

CoWERA provides:

- Binless WE resampling  
- GPU-accelerated execution  
- Native OpenMM integration  
- Full WE trajectory bookkeeping  
- Built-in rare-event kinetics analysis  

---
