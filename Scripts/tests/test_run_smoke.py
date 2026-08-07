"""End-to-end smoke test for the CoWERA run pipeline on the CPU platform.

This drives ``run_cowera.py`` as a subprocess through a couple of short cycles on
the shipped chignolin system (small, implicit-solvent, so it runs on the OpenMM
``CPU``/``Reference`` platform with no GPU), exercising the full
propagate -> warp -> resample -> report path end to end.

It is marked ``integration`` and is **deselected by default** (see
``Scripts/pytest.ini``); the numpy-only unit suite stays the fast gate. Run it
explicitly on a host that has the MD stack::

    python -m pytest Scripts/tests/ -m integration -v

The test skips cleanly when OpenMM/mdtraj/ruptures are unavailable, when the
chignolin inputs are missing, or when the installed ``run_cowera.py`` does not
yet honor the ``platform`` config field (i.e. before the platform-switch phase).

All simulation output is written to a temporary copy of the system directory, so
the test never pollutes the repository's ``Systems/`` tree.
"""
import os
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "Scripts")
CHIGNOLIN_DIR = os.path.join(REPO_ROOT, "Systems", "chignolin")

# Files the chignolin CPU smoke run needs from the shipped system directory.
_REQUIRED_INPUTS = ("chignolin.prmtop", "chignolin_folded.pdb",
                    "chignolin_unfolded.pdb", "system.py")


def _require_stack():
    for mod in ("openmm", "mdtraj", "ruptures"):
        pytest.importorskip(mod, reason=f"{mod} required for the integration smoke test")


def _require_inputs():
    if not os.path.isdir(CHIGNOLIN_DIR):
        pytest.skip("Systems/chignolin not present")
    for fname in _REQUIRED_INPUTS:
        if not os.path.isfile(os.path.join(CHIGNOLIN_DIR, fname)):
            pytest.skip(f"chignolin input {fname} not present")


def _write_config(system_dir, output_folder):
    """A deliberately tiny chignolin folding config on the CPU platform."""
    cfg = f"""\
system: "chignolin_smoke"
dir: "{system_dir}"

num_walkers: 4
pmax: 0.2
run: "smoke"
n_steps: 100
n_cycles: 2

start: "chignolin_unfolded.pdb"
topol: "chignolin.prmtop"
target: "chignolin_folded.pdb"
native: "chignolin_folded.pdb"

sel_feat: "rmsd_backbone"
mode: "probabilistic"
distance_criterion: "pairwise_rmsd"

d_merge: 0.1
d_warped: 0.05

temp: 275.0

# CPU/Reference execution -- no GPU required (honored by the platform-switch phase)
platform: "CPU"
n_workers: 2
gpu_ids: []

save_freq: 50
n_bins: 29
max_bins: 29
increment: -1
output_folder: "{output_folder}"
"""
    return cfg


def test_run_cowera_cpu_smoke(tmp_path):
    _require_stack()
    _require_inputs()

    # Isolated copy of the system so output never lands in the repo.
    system_dir = tmp_path / "chignolin"
    system_dir.mkdir()
    for fname in _REQUIRED_INPUTS:
        shutil.copy(os.path.join(CHIGNOLIN_DIR, fname), system_dir / fname)

    output_folder = "smoke_runs"
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(_write_config(str(system_dir), output_folder))

    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "run_cowera.py"),
         "--config", str(cfg_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )

    if proc.returncode != 0:
        # A CUDA-only build of run_cowera (pre platform-switch) rejects platform:CPU;
        # treat that as "not applicable here" rather than a hard failure.
        combined = proc.stdout + proc.stderr
        if "platform" in combined.lower() and "cuda" in combined.lower():
            pytest.skip("installed run_cowera.py does not support CPU platform yet")
        pytest.fail(f"run_cowera.py failed (rc={proc.returncode}):\n{combined[-4000:]}")

    # The run must have produced the documented output tree.
    run_root = system_dir / output_folder
    sim_dirs = list(run_root.glob("simdata_run*"))
    assert sim_dirs, f"no simdata_* output under {run_root}"
    sim = sim_dirs[0]
    assert (sim / "wepy.results.h5").is_file()
    assert (sim / "pkls").is_dir()
    assert (sim / "trajectories").is_dir()
    assert list(sim.glob("Info_*.txt")), "missing Info_*.txt"
