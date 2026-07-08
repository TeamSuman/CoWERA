# -*- coding: utf-8 -*-

import os, sys
import os.path as osp
import shutil
import time

sys.path.append(os.path.abspath("Scripts/"))
sys.path.append(os.path.abspath("Systems/"))

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

import warnings
warnings.filterwarnings("ignore")

from wepy.runners.openmm import OpenMMGPUWalkerTaskProcess, OpenMMCPUWalkerTaskProcess, OpenMMRunner, OpenMMWalker, OpenMMState, gen_sim_state
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.work_mapper.task_mapper import TaskMapper
from wepy.util.mdtraj import mdtraj_to_json_topology

from walker_pkl_reporter import WalkersPickleReporter

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
    args_cli = parser.parse_args()

    if args_cli.config:
        # Load YAML file
        with open(args_cli.config, "r") as f:
            cfg = yaml.safe_load(f)

        # Convert dict to argparse.Namespace for compatibility
        args = argparse.Namespace(**cfg)
    else:
        raise ValueError("Please provide a configuration file using --config")

    # ``gpu_ids`` is optional for CPU/Reference execution; default to an empty
    # list so len()/derived parameters stay well-defined.
    if not hasattr(args, "gpu_ids") or args.gpu_ids is None:
        args.gpu_ids = []

    # Derived parameters (same as before)
    args.n_gpu = len(args.gpu_ids)
    args.n_d = args.n_steps // args.save_freq

    return args


# Optional: colored terminal output
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    os.system("pip install colorama")
    from colorama import Fore, Style, init
    init(autoreset=True)

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
platform_name  = getattr(args, "platform", "CUDA")
platform_kwargs = getattr(args, "platform_kwargs", None)
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

    # If the folder exists, rename it with a timestamp
    if os.path.exists(outputs_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{outputs_dir}_backup_{timestamp}"
        shutil.move(outputs_dir, backup_name)
        os.makedirs(outputs_dir)
    else:
        os.makedirs(outputs_dir)
    #os.makedirs(outputs_dir, exist_ok=True)

    info_file_path = f'{outputs_dir}/Info_{run}.txt'

    # If the file exists, delete it
    if os.path.exists(info_file_path):
        os.remove(info_file_path)

    # Create a new empty file
    open(info_file_path, 'w').close()

    #select folder for dcd files
    dcd_folder = f'{outputs_dir}/trajectories/'

    os.makedirs(dcd_folder, exist_ok=True)


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

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, top.topology, integrator, platform=platform_name,
                          platform_kwargs=platform_kwargs, dcd_folder=dcd_folder, save_freq=save_freq)

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
                              use_cv_history=use_cv_history,
                              history_window=history_window)

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

    #output_dir = '/home/suman/Shaheerah/WeTICA/Systems/TC10b Trp-cage/simdata_run0_steps10000_cycs20000'
    #os.makedirs(output_dir, exist_ok=True)
    # Set up the HDF5 reporter
    hdf5_reporter = WepyHDF5Reporter(save_fields=('positions','box_vectors'),
                                file_path=osp.join(outputs_dir,f'wepy.results.h5') ,
                                resampler=resampler,
                                boundary_conditions=tbc,
                                topology=json_top)

    # Set up the pickle reporter (Essential for restarts)
    out_folder_pkl = osp.join(outputs_dir,f'pkls')
    pkl_reporter = WalkersPickleReporter(save_dir = out_folder_pkl,
                                      freq = 1,
                                      num_backups = 2)

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
        mapper = TaskMapper(walker_task_type=OpenMMGPUWalkerTaskProcess,
                            num_workers=num_workers,
                            platform=platform_name,
                            device_ids=gpu_ids)
    else:
        mapper = TaskMapper(walker_task_type=OpenMMCPUWalkerTaskProcess,
                            num_workers=num_workers,
                            platform=platform_name)


    # Build the simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=tbc,
                          work_mapper=mapper,
                          reporters=[hdf5_reporter, pkl_reporter, dashboard_reporter],
                          n_bins=n_bins,
                          max_bins = max_bins,
                          outputs_dir=outputs_dir
                          )






    #------------------------------
    # Run the simulation
    #------------------------------
    print(f"{Fore.YELLOW}{Style.BRIGHT}>>> Running the simulation...\n")
    # run a simulation with the manager for 'n_cycles' with 'n_steps' of integrator steps in each
    steps_list = [n_steps for i in range(n_cycles)]


    # and..... go!
    sim_manager.run_simulation(n_cycles,
                                steps_list)

    print(f"{Fore.GREEN}{Style.BRIGHT}\n✅ Simulation complete! Results saved in:\n{Fore.WHITE}{outputs_dir}\n")
    print(f"{Fore.CYAN}{'=' * 62}{Style.RESET_ALL}")
