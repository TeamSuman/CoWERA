# -*- coding: utf-8 -*-
"""Entry point for committor-guided CoWERA (opt-in extension).

Mirrors ``run_cowera.py`` but swaps the geometric progress coordinate for an
on-the-fly-trained committor: it builds an ``MLPCommittor`` + featurizer, a
``BootstrapPhaseManager``, and a ``CommittorResampler`` (which subclasses the
standard ``CoWERAResampler``, so the MD/WE machinery is unchanged).

Requires the full simulation stack (OpenMM, mdtraj, MDAnalysis) and a GPU; run
on the simulation host. See ``docs/COMMITTOR_EXTENSION.md`` and
``config_committor.yml``.
"""
import os
import sys
import os.path as osp
import shutil
import argparse
from datetime import datetime

sys.path.append(os.path.abspath("Scripts/"))
sys.path.append(os.path.abspath("Systems/"))

import yaml
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import simtk.openmm.app as omma
import simtk.unit as unit
import mdtraj as mdj

from wepy.runners.openmm import (
    OpenMMGPUWalkerTaskProcess, OpenMMRunner, OpenMMWalker, OpenMMState, gen_sim_state)
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.work_mapper.task_mapper import TaskMapper
from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.openmm import OpenMMRunnerDashboardSection

from walker_pkl_reporter import WalkersPickleReporter
from cowera.warper import TargetBC
from cowera.sim_manager import Manager
from cowera.features import best_hummer_q, get_native_contacts

from cowera_committor.featurizer import DistanceFeaturizer
from cowera_committor.committor_model import MLPCommittor
from cowera_committor.phase_manager import BootstrapPhaseManager
from cowera_committor.committor_resampler import CommittorResampler


def get_args():
    parser = argparse.ArgumentParser(description="Run committor-guided CoWERA.")
    parser.add_argument("--config", type=str, required=True, help="YAML config.")
    cli = parser.parse_args()
    with open(cli.config) as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)
    args.n_gpu = len(args.gpu_ids)
    args.n_d = args.n_steps // args.save_freq
    return args


def build_featurizer(cfg, native_path):
    """Build the committor featurizer from the native structure."""
    if cfg.get("featurizer", "native_contacts") == "native_contacts":
        contacts, _ = get_native_contacts(
            native_path, native_cutoff=cfg.get("contact_cutoff", 0.45))
        return DistanceFeaturizer(np.asarray(contacts)), len(contacts)
    raise ValueError(f"Unknown featurizer: {cfg.get('featurizer')}")


def main():
    args = get_args()
    cfg = args.committor
    assert cfg.get("enabled", False), "committor.enabled must be true for this entry point"

    inp = args.dir
    start_path = f"{inp}/{args.start}"
    native_path = f"{inp}/{args.native}"
    top_path = f"{inp}/{args.topol}"
    tar_path = f"{inp}/{args.target}" if args.target else None

    outputs_dir = (f"{inp}/{args.output_folder}/"
                   f"simdata_run{args.run}_steps{args.n_steps}_cycs{args.n_cycles}")
    if os.path.exists(outputs_dir):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(outputs_dir, f"{outputs_dir}_backup_{ts}")
    os.makedirs(outputs_dir)
    info_file_path = f"{outputs_dir}/Info_{args.run}.txt"
    open(info_file_path, "w").close()
    dcd_folder = f"{outputs_dir}/trajectories/"
    os.makedirs(dcd_folder, exist_ok=True)

    # --- OpenMM system (same as run_cowera.py) ---
    import importlib.util
    spec = importlib.util.spec_from_file_location("system", os.path.join(inp, "system.py"))
    system_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(system_module)

    coord_file = omma.GromacsGroFile(start_path)
    box_vectors = coord_file.getPeriodicBoxVectors()
    top = omma.GromacsTopFile(top_path, periodicBoxVectors=box_vectors)
    system, integrator = system_module.make_system(top, args.temp)
    new_simtk_state = gen_sim_state(coord_file.getPositions(), system, integrator)
    runner = OpenMMRunner(system, top.topology, integrator, platform="CUDA",
                          dcd_folder=dcd_folder, save_freq=args.save_freq)

    walker_state = OpenMMState(new_simtk_state)
    init_walkers = [OpenMMWalker(walker_state, 1.0 / args.num_walkers)
                    for _ in range(args.num_walkers)]
    json_top = mdtraj_to_json_topology(mdj.load(start_path).top)

    # target position for the warper (committor target = B = q -> 1)
    target_pos, _ = best_hummer_q(traj=tar_path, native_file=native_path)

    # --- committor pieces ---
    featurizer, input_dim = build_featurizer(cfg, native_path)
    model = MLPCommittor(input_dim=input_dim,
                         hidden=tuple(cfg.get("hidden_layers", [64, 64])),
                         l2=cfg.get("l2", 1e-5))
    phase_manager = BootstrapPhaseManager(
        warmup_cycles=cfg.get("warmup_cycles", 20),
        retrain_interval=cfg.get("retrain_interval", 10),
        a_cutoff=cfg.get("basin_a_cutoff", 0.1),
        b_cutoff=cfg.get("basin_b_cutoff", 0.9),
        var_threshold=cfg.get("var_threshold", 0.05),
        min_intermediate_pairs=cfg.get("min_intermediate_pairs", 50),
        hist_mean_tol=cfg.get("hist_mean_tol", 0.1),
        hist_std_tol=cfg.get("hist_std_tol", 0.15),
        required_consecutive=cfg.get("required_consecutive", 3),
    )

    resampler = CommittorResampler(
        committor_model=model, featurizer=featurizer, phase_manager=phase_manager,
        merge_alpha=cfg.get("merge_alpha", 0.7),
        distance_criterion=args.distance_criterion,
        native_file=native_path, top_file=native_path,
        retrain_epochs=cfg.get("retrain_epochs", 200),
        retrain_lr=cfg.get("learning_rate", 5e-3),
        semigroup_buffer_size=cfg.get("semigroup_buffer_size", 10000),
        shooting_buffer_size=cfg.get("shooting_buffer_size", 2000),
        increment=args.increment,
        init_state=walker_state, merge_dist=args.d_merge, run_id=args.run,
        info_file_path=info_file_path, n_d=args.n_d, dcd_folder=dcd_folder,
        pmax=args.pmax, mode=args.mode,
    )

    tbc = TargetBC(cutoff_distance=args.d_warped, initial_state=walker_state,
                   target_pos=target_pos, feat="best_hummer_q", dcd_folder=dcd_folder,
                   native_file=native_path, init_file=start_path, tar_file=tar_path,
                   top_file=native_path)

    hdf5_reporter = WepyHDF5Reporter(
        save_fields=("positions", "box_vectors"),
        file_path=osp.join(outputs_dir, "wepy.results.h5"),
        resampler=resampler, boundary_conditions=tbc, topology=json_top)
    pkl_reporter = WalkersPickleReporter(
        save_dir=osp.join(outputs_dir, "pkls"), freq=1, num_backups=2)
    dashboard_reporter = DashboardReporter(
        file_path=osp.join(outputs_dir, "wepy.dash.org"),
        runner_dash=OpenMMRunnerDashboardSection(runner))

    mapper = TaskMapper(walker_task_type=OpenMMGPUWalkerTaskProcess,
                        num_workers=args.n_gpu, platform="CUDA", device_ids=args.gpu_ids)

    sim_manager = Manager(
        init_walkers, runner=runner, resampler=resampler, boundary_conditions=tbc,
        work_mapper=mapper,
        reporters=[hdf5_reporter, pkl_reporter, dashboard_reporter],
        n_bins=args.n_bins, max_bins=args.max_bins, outputs_dir=outputs_dir)

    print(f">>> Running committor-guided CoWERA ({args.num_walkers} walkers)...")
    sim_manager.run_simulation(args.n_cycles, [args.n_steps] * args.n_cycles)
    print(f"✅ Done. Results in {outputs_dir}")


if __name__ == "__main__":
    main()
