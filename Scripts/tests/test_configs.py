"""Doc-lint: every shipped Systems/<name>/config.yml must be loadable and its
referenced input files must exist, so the documented examples actually run.

Pure filesystem + YAML -- no MD stack required.
"""
import glob
import os

import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SYSTEMS_DIR = os.path.join(REPO_ROOT, "Systems")

CONFIGS = sorted(glob.glob(os.path.join(SYSTEMS_DIR, "*", "config.yml")))

REQUIRED_KEYS = [
    "system", "dir", "num_walkers", "n_steps", "n_cycles",
    "start", "topol", "native", "sel_feat", "mode", "distance_criterion",
    "save_freq", "n_bins", "max_bins", "increment", "output_folder",
]


@pytest.mark.skipif(not CONFIGS, reason="no shipped Systems/*/config.yml found")
@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_shipped_config_is_valid_and_runnable(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # required keys present
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    assert not missing, f"{config_path} missing keys: {missing}"

    # feature / metric / mode take known values
    assert cfg["sel_feat"] in ("best_hummer_q", "rmsd_backbone")
    assert cfg["mode"] in ("greedy", "probabilistic")
    assert cfg["distance_criterion"] in ("pairwise_rmsd", "euclidean")
    assert cfg["increment"] in (1, -1)

    # rmsd_backbone requires a target (used to define the RMSD range)
    if cfg["sel_feat"] == "rmsd_backbone":
        assert cfg.get("target") is not None, "rmsd_backbone requires 'target'"

    # the system directory and its inputs must exist (dir is relative to repo root)
    sys_dir = os.path.normpath(os.path.join(REPO_ROOT, cfg["dir"]))
    assert os.path.isdir(sys_dir), f"dir does not exist: {sys_dir}"
    assert os.path.isfile(os.path.join(sys_dir, "system.py")), f"missing system.py in {sys_dir}"

    referenced = [cfg["start"], cfg["topol"], cfg["native"]]
    if cfg.get("target") is not None:
        referenced.append(cfg["target"])
    for fname in referenced:
        assert os.path.isfile(os.path.join(sys_dir, fname)), \
            f"{config_path}: referenced file '{fname}' not found in {sys_dir}"

    # derived n_d must be >= 1 (n_steps must be a multiple-ish of save_freq)
    assert cfg["n_steps"] // cfg["save_freq"] >= 1, "n_steps // save_freq must be >= 1"
