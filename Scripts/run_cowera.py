# -*- coding: utf-8 -*-

import os, sys
import os.path as osp
import shutil
import time
import pickle

sys.path.append(os.path.abspath("Scripts/"))
sys.path.append(os.path.abspath("Systems/"))

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

import warnings
# Silence only the noisy third-party deprecation/future chatter, NOT UserWarnings.
# CoWERA's feature functions surface trajectory-read fallbacks (which substitute the
# initial structure's CV and can bias resampling) via logging.warning -> dashboard.log;
# a blanket filterwarnings("ignore") previously hid those entirely.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from wepy.runners.openmm import OpenMMGPUWalkerTaskProcess, OpenMMCPUWalkerTaskProcess, OpenMMGPUWorker, OpenMMRunner, OpenMMWalker, OpenMMState, gen_sim_state
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.work_mapper.task_mapper import TaskMapper
from wepy.work_mapper.mapper import WorkerMapper
from wepy.util.mdtraj import mdtraj_to_json_topology

from walker_pkl_reporter import WalkersPickleReporter, latest_checkpoint

from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.openmm import OpenMMRunnerDashboardSection
import logging

import numpy as np
from datetime import datetime

from cowera.features import best_hummer_q, RMSD_Backbone
from cowera.metric import Calculate_Distances
from cowera.resampler import CoWERAResampler
from cowera.warper import TargetBC
from cowera.sim_manager import Manager

import yaml
import argparse
import importlib.util
#os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = "50"
# nvidia-cuda-mps-control -d
#export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

def get_args():
    parser = argparse.ArgumentParser(description="Run CoWERA simulation.")
    parser.add_argument("--config", type=str, help="Path to YAML input file.")
    parser.add_argument("--restart", action="store_true",
                        help="Resume from the newest checkpoint in the output "
                             "directory instead of starting a fresh run.")
    args_cli = parser.parse_args()

    if args_cli.config:
        # Load YAML file
        with open(args_cli.config, "r") as f:
            cfg = yaml.safe_load(f)

        # Convert dict to argparse.Namespace for compatibility
        args = argparse.Namespace(**cfg)
    else:
        raise ValueError("Please provide a configuration file using --config")

    # Restart may be requested on the CLI or in the config file.
    args.restart = bool(args_cli.restart or getattr(args, "restart", False))

    # ``gpu_ids`` is optional for CPU/Reference execution; default to an empty
    # list so len()/derived parameters stay well-defined.
    if not hasattr(args, "gpu_ids") or args.gpu_ids is None:
        args.gpu_ids = []

    # Derived parameters (same as before)
    args.n_gpu = len(args.gpu_ids)
    args.n_d = args.n_steps // args.save_freq

    return args


# Optional: colored terminal output. Never install packages at runtime (compute
# nodes are often offline / read-only); fall back to a no-op color shim instead.
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class _NoColor:
        def __getattr__(self, _name):
            return ""
    Fore = Style = _NoColor()
    def init(*args, **kwargs):
        pass

# -------------------- Parse args -------------------- #
args = get_args()

# -------------------- Initialize variables -------------------- #
system         = args.system
DIR            = args.dir
num_walkers    = args.num_walkers
run            = args.run
n_steps        = args.n_steps
n_cycles       = args.n_cycles
start          = args.start
topol          = args.topol
target         = args.target
native        = args.native
sel_feat       = args.sel_feat
d_merge        = args.d_merge
d_warped       = args.d_warped
temp           = args.temp
gpu_ids        = args.gpu_ids
n_gpu          = args.n_gpu
save_freq      = args.save_freq
n_d            = args.n_d
n_bins         = args.n_bins
max_bins       = args.max_bins
increment      = args.increment
output_folder  = args.output_folder
pmax           = args.pmax
mode           = args.mode
distance_criterion = args.distance_criterion

# Compute platform selection (default CUDA preserves prior behaviour). Non-CUDA
# platforms (CPU/Reference/OpenCL) let CoWERA run on CPU-only HPC nodes.
restart        = args.restart
checkpoint_freq = getattr(args, "checkpoint_freq", 10)
scratch_dir    = getattr(args, "scratch_dir", None)
seed           = getattr(args, "seed", None)
deterministic_dynamics = getattr(args, "deterministic_dynamics", False)
# Reporters save only positions + box_vectors; velocities are retained for state
# fidelity across clone/merge/warp. Forces/energy/parameters/derivatives are
# fetched every segment but discarded, so trim them by default to cut GPU->CPU
# transfer. Config `getstate_kwargs` can override (e.g. to keep forces).
_DEFAULT_GETSTATE = {'getPositions': True, 'getVelocities': True,
                     'getForces': False, 'getEnergy': False,
                     'getParameters': False, 'getParameterDerivatives': False}
getstate_kwargs = getattr(args, "getstate_kwargs", None) or _DEFAULT_GETSTATE
platform_name  = getattr(args, "platform", "CUDA")
platform_kwargs = getattr(args, "platform_kwargs", None)
# EXPERIMENTAL (default off): persistent per-GPU workers with a cached OpenMM
# Context, instead of forking a process + rebuilding the Context every walker
# every cycle. Big multi-GPU speedup, but must be validated on a GPU host.
persistent_workers = getattr(args, "persistent_workers", False)
_GPU_PLATFORMS = ("CUDA", "OpenCL")
is_gpu_platform = platform_name in _GPU_PLATFORMS
# Number of concurrent walker worker processes. For GPU platforms this is the
# number of device slots (len(gpu_ids), repeated ids => MPS packing); for CPU it
# is an explicit ``n_workers`` (default: 1, or n_gpu if that was set).
if is_gpu_platform:
    num_workers = n_gpu
else:
    num_workers = getattr(args, "n_workers", None) or (n_gpu if n_gpu > 0 else 1)

# Determine folding or unfolding and set target behavior

if sel_feat == "best_hummer_q":

    # Determine folding direction
    fold_state = "Folding" if increment == 1 else "Unfolding"


elif sel_feat == "rmsd_backbone":

    # Target is mandatory for RMSD
    if target is None:
        raise ValueError(
            "target_file must be provided for computing rmsd range."
        )

    # Folding defined oppositely for RMSD
    fold_state = "Folding" if increment == -1 else "Unfolding"


else:
    raise ValueError(f"Unsupported feature type: {sel_feat}")

# -------------------- Flashy terminal banner -------------------- #
def flashy_banner():
    from datetime import datetime
    from colorama import Fore, Style

    # Simple ASCII banner — no exotic characters
    banner = (
        f"{Fore.CYAN}{Style.BRIGHT}\n"
        "==============================================================\n"
        "           CoWERA Simulation Launchpad\n"
        "==============================================================\n"
        f"{Style.RESET_ALL}"
    )


    print(banner)
    print(f"{Fore.YELLOW}{Style.BRIGHT}>>> Simulation initialized ...\n")


    # Print all inputs clearly
    print(f"{Fore.MAGENTA}{Style.BRIGHT}System:           {Fore.WHITE}{system}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Directory:        {Fore.WHITE}{DIR}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Run ID:           {Fore.WHITE}{run}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Process:             {Fore.GREEN}{fold_state}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Mode:             {Fore.WHITE}{mode}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Temperature:      {Fore.WHITE}{temp} K")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Walkers:          {Fore.WHITE}{num_walkers}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Max Walker Prob:  {Fore.WHITE}{pmax}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Platform:         {Fore.WHITE}{platform_name}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Workers:          {Fore.WHITE}{num_workers}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}GPUs:             {Fore.WHITE}{gpu_ids}  (Total: {n_gpu})")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Steps per cycle:  {Fore.WHITE}{n_steps}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Total cycles:     {Fore.WHITE}{n_cycles}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Save frequency:   {Fore.WHITE}{save_freq}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}n_d (segments):   {Fore.WHITE}{n_d}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Feature:          {Fore.WHITE}{sel_feat}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Distance Criterion: {Fore.WHITE}{distance_criterion}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}d_merge:          {Fore.WHITE}{d_merge}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}d_warped:         {Fore.WHITE}{d_warped}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Bins:             {Fore.WHITE}{n_bins}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Max bins:         {Fore.WHITE}{max_bins}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Start file:       {Fore.WHITE}{start}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Topology file:    {Fore.WHITE}{topol}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Target file:      {Fore.WHITE}{target}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Reference GRO:    {Fore.WHITE}{native}")
    print(f"{Fore.CYAN}{'=' * 62}{Style.RESET_ALL}\n")

# -------------------- Launch -------------------- #

if __name__ == "__main__":
    flashy_banner()

    inp_path = DIR
    start_path = f'{inp_path}/{start}'
    native_path = f'{inp_path}/{native}'
    top_path = f'{inp_path}/{topol}'
    if target is not None:
        tar_path = f'{inp_path}/{target}'
        print(f"target path: {tar_path}")


    outputs_dir = f'{inp_path}/{output_folder}/simdata_run{run}_steps{n_steps}_cycs{n_cycles}'

    if restart:
        # Resume: reuse the existing output directory in place. Never move it
        # aside or clear it, so trajectories, results and checkpoints survive.
        if not os.path.exists(outputs_dir):
            raise FileNotFoundError(
                f"--restart given but no existing output directory to resume: {outputs_dir}"
            )
        print(f"{Fore.YELLOW}{Style.BRIGHT}>>> RESTART: resuming run in {outputs_dir}")
    else:
        # Fresh run: if the folder exists, rename it with a timestamp backup.
        if os.path.exists(outputs_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{outputs_dir}_backup_{timestamp}"
            shutil.move(outputs_dir, backup_name)
        os.makedirs(outputs_dir)

    info_file_path = f'{outputs_dir}/Info_{run}.txt'

    # On a fresh run start with an empty Info file; on restart append to the
    # existing one so the per-cycle log stays contiguous across the checkpoint.
    if not restart:
        if os.path.exists(info_file_path):
            os.remove(info_file_path)
        open(info_file_path, 'w').close()
    elif not os.path.exists(info_file_path):
        open(info_file_path, 'w').close()

    #select folder for dcd files
    final_dcd_folder = f'{outputs_dir}/trajectories/'
    os.makedirs(final_dcd_folder, exist_ok=True)

    if scratch_dir:
        # Node-local scratch: the many small per-cycle DCD read/append/copy/remove
        # operations run on fast local disk instead of a shared parallel filesystem
        # (Lustre/GPFS), which they would otherwise hammer with metadata ops. The
        # durable checkpoints (pkls/) and results (h5) stay on the output dir so a
        # restart still works. Trajectories are synced back on exit.
        dcd_folder = osp.join(scratch_dir,
                              f'cowera_run{run}_steps{n_steps}_cycs{n_cycles}',
                              'trajectories') + os.sep
        os.makedirs(dcd_folder, exist_ok=True)

        # On restart, re-stage any previously synced trajectories to scratch so the
        # in-memory CV histories can rebuild from them.
        if restart:
            for fname in os.listdir(final_dcd_folder):
                shutil.copy(osp.join(final_dcd_folder, fname), osp.join(dcd_folder, fname))

        def _sync_trajectories_back():
            try:
                os.makedirs(final_dcd_folder, exist_ok=True)
                for fname in os.listdir(dcd_folder):
                    shutil.copy(osp.join(dcd_folder, fname), osp.join(final_dcd_folder, fname))
                print(f"Synced trajectories from scratch -> {final_dcd_folder}")
            except Exception as exc:
                print(f"Warning: failed to sync trajectories from scratch ({exc}).")

        import atexit
        atexit.register(_sync_trajectories_back)
        print(f"{Fore.CYAN}Trajectories staged on node-local scratch: {dcd_folder}")
    else:
        dcd_folder = final_dcd_folder


    system_file = os.path.join(inp_path, "system.py")

    if not os.path.isfile(system_file):
        raise FileNotFoundError(f"{system_file} not found")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("system", system_file)
    system_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(system_module)

    # Import make_system
    make_system = system_module.make_system




    # ----------- set up OpenMM system and runner ----------- #

    # Load atoms and topology objects
    start_ext = os.path.splitext(start_path)[1].lower()
    top_ext = os.path.splitext(top_path)[1].lower()

    # --------------------------
    # Load coordinates
    # --------------------------
    if start_ext == ".gro":
        coord_file = omma.GromacsGroFile(start_path)
        box_vectors = coord_file.getPeriodicBoxVectors()

    elif start_ext == ".pdb":
        coord_file = omma.PDBFile(start_path)
        box_vectors = coord_file.topology.getPeriodicBoxVectors()

    else:
        raise ValueError("start_path must be .gro or .pdb")

    # --------------------------
    # Load topology
    # --------------------------
    if top_ext == ".top":
        # GROMACS topology
        top = omma.GromacsTopFile(
            top_path,
            periodicBoxVectors=box_vectors
        )

    elif top_ext == ".prmtop":
        # AMBER topology
        top = omma.AmberPrmtopFile(top_path)

    else:
        raise ValueError("top_path must be .top, .prmtop")



    # Get positions from gro file
    pos = coord_file.getPositions()


    # Make system
    # system = top.createSystem(nonbondedMethod=omma.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=omma.HBonds)
    # system.addForce(omm.openmm.MonteCarloBarostat(1*unit.bar, temp*unit.kelvin)) # NPT ensemble
    # integrator = omm.openmm.LangevinMiddleIntegrator(temp*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)

    system, integrator = make_system(top, temp)

    # Generate a new simtk "state"
    new_simtk_state = gen_sim_state(pos, system, integrator)
    #target_state = gen_sim_state(tar_pos, system, integrator)

    # set up the OpenMMRunner with your system. random_seed (only when
    # deterministic_dynamics is requested) makes the per-segment integrator seed a
    # deterministic function of (seed, cycle, walker) instead of OpenMM's default 0
    # (which re-randomizes each segment).
    # Context reuse only helps with a persistent mapper (WorkerMapper); with the
    # default process-per-task mapper each process dies after one segment.
    use_persistent = persistent_workers and is_gpu_platform
    runner = OpenMMRunner(system, top.topology, integrator, platform=platform_name,
                          platform_kwargs=platform_kwargs, dcd_folder=dcd_folder, save_freq=save_freq,
                          random_seed=(seed if deterministic_dynamics else None),
                          getState_kwargs=getstate_kwargs,
                          reuse_context=use_persistent)

    # Select the feature
    sel_feat = sel_feat

    print(f"Selected feature: {sel_feat}")
    print(f"native path: {native_path}")
    print(f"start path: {start_path}")
    print(f"topology path: {top_path}")
    print(f"target path: {tar_path}")

    if sel_feat == "best_hummer_q":
        init_pos, d_range = best_hummer_q(
            traj=start_path,
            native_file=native_path,
        )
        if target is not None:
            target_pos, _ = best_hummer_q(
                traj=tar_path,
                native_file=native_path,
            )
        else:
            target_pos = 1 if fold_state == "Folding" else 0

    elif sel_feat == "rmsd_backbone":

        if increment == 1:
            init_pos, d_range = RMSD_Backbone(
            traj=start_path,
            top_file=native_path,
            init_file=start_path,
            unfolded_file=tar_path
        )
            target_pos, _ = RMSD_Backbone(
                traj=tar_path,
                top_file=native_path,
                init_file=start_path,
                unfolded_file=tar_path
            )
        else:
            init_pos, d_range = RMSD_Backbone(
            traj=start_path,
            top_file=native_path,
            init_file=start_path,
            unfolded_file=start_path
        )
            target_pos, _ = RMSD_Backbone(
                traj=tar_path,
                top_file=native_path,
                init_file=start_path,
                unfolded_file=start_path
            )

    print(f"Initial position: {init_pos}, Target position: {target_pos}, d_range: {d_range}")







    # -------------------- Create walker state objects -------------------- #
    print('Creating walker state objects...')

    # Get the walker topology in a json format
    json_top = mdtraj_to_json_topology(mdj.load(start_path).top)

    # Set up parameters for running the simulation
    init_weight = 1.0 / num_walkers
    # Generate the walker state in wepy format
    walker_state = OpenMMState(new_simtk_state)
    #target_walker_state = OpenMMState(target_state)

    # --------------------- Restart bookkeeping --------------------- #
    # Defaults for a fresh run; overwritten below when resuming.
    start_cycle = 0
    continue_run = None
    restored_n_bins = None
    pkl_dir = osp.join(outputs_dir, 'pkls')

    if restart:
        ckpt = latest_checkpoint(pkl_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f"--restart given but no complete checkpoint found in {pkl_dir}"
            )
        walkers_path, meta_path, ckpt_cycle = ckpt
        with open(walkers_path, 'rb') as f:
            init_walkers = pickle.load(f)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        start_cycle = ckpt_cycle + 1
        restored_n_bins = meta.get('n_bins')
        print(f"{Fore.YELLOW}{Style.BRIGHT}>>> Resuming from cycle {ckpt_cycle} "
              f"(next cycle {start_cycle}), n_bins={restored_n_bins}")
        if start_cycle >= n_cycles:
            print(f"{Fore.GREEN}{Style.BRIGHT}Checkpoint already at/after n_cycles; nothing to do.")
            sys.exit(0)
    else:
        # Make a list of the initial walkers
        init_walkers = [OpenMMWalker(walker_state, init_weight) for i in range(num_walkers)]


    # Distance metric to be used in resampling
    proj_distance = Calculate_Distances(sel_feat, increment=increment, native_file=native_path, init_file=start_path,
                                        tar_file=tar_path, top_file=native_path, distance_criterion=distance_criterion)

    #init_rmsd = proj_distance.image_distance(walker_state, target_walker_state)
    #print(f"Initial rmsd distance from target: {init_rmsd}")











    #------------------------------------------------------------------------------
    # Building wepy objects
    #-------------------------------------------------------------------------------
    print('Creating the wepy objects...')



    # Phase 2 in-memory CV history options (optional in config; sensible
    # defaults keep existing config files working unchanged).
    use_cv_history = getattr(args, "use_cv_history", True)
    history_window = getattr(args, "history_window", None)
    # Adaptive-bin adjustment factors (paper Appendix F). Defaults preserve the
    # existing code behaviour (1.2 / 0.8). NOTE (decision D-1): the manuscript
    # states 1.5 / 0.75 -- set these explicitly to reproduce the paper text.
    bin_increase_factor = getattr(args, "bin_increase_factor", 1.2)
    bin_decrease_factor = getattr(args, "bin_decrease_factor", 0.8)

    # Set up the Resampler with the parameters
    resampler = CoWERAResampler(distance=proj_distance,
                              init_state=walker_state,
                              merge_dist=d_merge,
                              run_id=run,
                              info_file_path=info_file_path,
                              increment=increment,
                              n_d=n_d,
                              dcd_folder=dcd_folder,
                              pmax=pmax,
                              mode=mode,
                              seed=seed,
                              use_cv_history=use_cv_history,
                              history_window=history_window,
                              bin_increase_factor=bin_increase_factor,
                              bin_decrease_factor=bin_decrease_factor)

    # Set up the boundary conditions for a non-eq ensemble
    tbc = TargetBC(cutoff_distance=d_warped,
                    initial_state=walker_state,
                    target_pos=target_pos,
                    feat=sel_feat,
                    dcd_folder=dcd_folder,
                    native_file=native_path,
                    init_file=start_path,
                    tar_file=tar_path,
                    top_file=native_path
                    )

    # Set up the HDF5 reporter. A fresh run creates the file (mode 'x'); a
    # restart opens the existing file (mode 'r+') and links a new run as a
    # continuation of the last one (continue_run) so analysis can follow it.
    h5_path = osp.join(outputs_dir, 'wepy.results.h5')
    hdf5_mode = 'x'
    if restart:
        hdf5_mode = 'r+'
        try:
            from wepy.hdf5 import WepyHDF5
            with WepyHDF5(h5_path, mode='r') as _probe:
                run_idxs = list(_probe.run_idxs)
            continue_run = max(run_idxs) if run_idxs else None
        except Exception as exc:
            logging.warning(f"Could not read existing runs from {h5_path} ({exc}); "
                            "continuing as an unlinked run.")
            continue_run = None

    hdf5_reporter = WepyHDF5Reporter(save_fields=('positions','box_vectors'),
                                file_path=h5_path,
                                mode=hdf5_mode,
                                resampler=resampler,
                                boundary_conditions=tbc,
                                topology=json_top)

    # Set up the pickle/checkpoint reporter (enables --restart). Keep existing
    # checkpoints on restart so the one that seeded this run is not wiped.
    out_folder_pkl = pkl_dir
    pkl_reporter = WalkersPickleReporter(save_dir = out_folder_pkl,
                                      freq = checkpoint_freq,
                                      num_backups = 2,
                                      wipe_on_init = not restart)

    # Set up the dashboard reporter
    dashboard_path = osp.join(outputs_dir,f'wepy.dash.org')
    openmm_dashboard_sec = OpenMMRunnerDashboardSection(runner)
    dashboard_reporter = DashboardReporter(file_path = dashboard_path,
                                        runner_dash = openmm_dashboard_sec)


    # Create a work mapper. GPU platforms (CUDA/OpenCL) assign a device per worker
    # slot via DeviceIndex; CPU/Reference platforms use a device-agnostic task type.
    if is_gpu_platform:
        if not gpu_ids:
            raise ValueError(
                f"platform '{platform_name}' requires a non-empty 'gpu_ids' list in the config."
            )
        if use_persistent:
            # Persistent long-lived worker per device slot; the runner caches its
            # OpenMM Context so each segment is just setState -> step -> getState.
            print(f"{Fore.YELLOW}{Style.BRIGHT}>>> Using persistent per-GPU workers "
                  f"(cached Context) [EXPERIMENTAL]")
            mapper = WorkerMapper(worker_type=OpenMMGPUWorker,
                                  num_workers=num_workers,
                                  platform=platform_name,
                                  device_ids=gpu_ids)
        else:
            mapper = TaskMapper(walker_task_type=OpenMMGPUWalkerTaskProcess,
                                num_workers=num_workers,
                                platform=platform_name,
                                device_ids=gpu_ids)
    else:
        mapper = TaskMapper(walker_task_type=OpenMMCPUWalkerTaskProcess,
                            num_workers=num_workers,
                            platform=platform_name)


    # Build the simulation manager. On restart resume the adaptive bin count
    # snapshotted in the checkpoint so the bin resolution stays continuous.
    active_n_bins = restored_n_bins if (restart and restored_n_bins is not None) else n_bins
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=tbc,
                          work_mapper=mapper,
                          reporters=[hdf5_reporter, pkl_reporter, dashboard_reporter],
                          n_bins=active_n_bins,
                          max_bins = max_bins,
                          outputs_dir=outputs_dir
                          )






    #------------------------------
    # Run the simulation
    #------------------------------
    print(f"{Fore.YELLOW}{Style.BRIGHT}>>> Running the simulation...\n")
    # run a simulation with the manager for 'n_cycles' with 'n_steps' of integrator steps in each
    steps_list = [n_steps for i in range(n_cycles)]


    # and..... go!  (start_cycle/continue_run are non-trivial only on restart)
    sim_manager.run_simulation(n_cycles,
                                steps_list,
                                start_cycle=start_cycle,
                                continue_run=continue_run)

    print(f"{Fore.GREEN}{Style.BRIGHT}\n✅ Simulation complete! Results saved in:\n{Fore.WHITE}{outputs_dir}\n")
    print(f"{Fore.CYAN}{'=' * 62}{Style.RESET_ALL}")
