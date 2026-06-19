import warnings
from itertools import combinations

import numpy as np

# mdtraj is only required for the file/trajectory based code paths. Import it
# lazily so that the pure-numpy cores (e.g. ``compute_q``) remain importable and
# testable in environments without the heavy MD stack installed.
try:
    import mdtraj as md
except Exception:  # pragma: no cover - exercised only when mdtraj is absent
    md = None


def _require_mdtraj():
    if md is None:
        raise ImportError(
            "mdtraj is required for trajectory based feature calculations "
            "but is not installed."
        )


######################################## Best Hummer FNC ####################################################

#############################################################################################################

# Default smoothing/cutoff constants for the Best-Hummer fraction of native
# contacts (Q). Exposed as module constants so they can be reused/tested.
BETA_CONST = 50.0       # 1/nm
LAMBDA_CONST = 1.8
NATIVE_CUTOFF = 0.45    # nm
MIN_RES_SEP = 3         # minimum residue separation for a "native contact"

# Cache of native-contact definitions keyed by the reference structure file and
# the contact-definition parameters. Recomputing the heavy-atom pair
# combinations and the native-contact filter on every walker every cycle was a
# dominant CPU cost; it only depends on the (static) native structure.
_NATIVE_CONTACTS_CACHE = {}


def compute_q(r, r0, beta=BETA_CONST, lam=LAMBDA_CONST):
    """Best-Hummer fraction of native contacts from contact distances.

    Pure-numpy core (no mdtraj dependency) so it can be unit tested directly.

    Parameters
    ----------
    r : np.ndarray, shape (n_frames, n_contacts)
        Current distances for each native contact pair, per frame.
    r0 : np.ndarray, shape (1, n_contacts) or (n_contacts,)
        Reference (native) distances for each contact pair.
    beta, lam : float
        Smoothing parameters (Eq. A1 of the CoWERA paper).

    Returns
    -------
    q : np.ndarray, shape (n_frames,)
        The fraction of native contacts for each frame, in [0, 1].
    """
    r = np.atleast_2d(np.asarray(r, dtype=float))
    r0 = np.asarray(r0, dtype=float).reshape(1, -1)
    return np.mean(1.0 / (1.0 + np.exp(beta * (r - lam * r0))), axis=1)


def get_native_contacts(native_file, native_cutoff=NATIVE_CUTOFF, min_res_sep=MIN_RES_SEP):
    """Return (and cache) the native-contact atom pairs and reference distances.

    Parameters
    ----------
    native_file : str
        Path to the native/reference structure.

    Returns
    -------
    native_contacts : np.ndarray, shape (n_contacts, 2)
    r0 : np.ndarray, shape (1, n_contacts)
        Native contact distances (as returned by ``md.compute_distances``).
    """
    _require_mdtraj()

    key = (native_file, float(native_cutoff), int(min_res_sep))
    cached = _NATIVE_CONTACTS_CACHE.get(key)
    if cached is not None:
        return cached

    native = md.load(native_file)

    # Heavy-atom pairs separated by more than ``min_res_sep`` residues.
    heavy = native.topology.select_atom_indices('heavy')
    res_index = np.array([native.topology.atom(i).residue.index for i in heavy])

    heavy_pairs = np.array([
        (i, j) for (i, j) in combinations(heavy, 2)
        if abs(native.topology.atom(i).residue.index
               - native.topology.atom(j).residue.index) > min_res_sep
    ])

    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    native_contacts = heavy_pairs[heavy_pairs_distances < native_cutoff]
    r0 = md.compute_distances(native[0], native_contacts)

    result = (native_contacts, r0)
    _NATIVE_CONTACTS_CACHE[key] = result
    return result


def best_hummer_q(traj, native_file=None, init_file=None, tar_file=None, frame_range=None):
    """
    Compute the Best Hummer Q from a trajectory and native structure.

    Parameters:
    - traj: str, md.Trajectory, or np.ndarray
        The trajectory data. Can be a file path, an md.Trajectory object, or a numpy array of coordinates.
    - native_file: str
        Path to the native structure file.
    - frame_range: tuple (start, end), optional
        Range of frames to consider from the trajectory.
    - init_file: str, optional
        Initial structure file used as a single fallback if the primary
        trajectory cannot be read (e.g. a transiently half-written DCD file).

    Returns:
    - q: np.ndarray
        Q values for each frame
    - d_range: np.ndarray
        The [min, max] range of the coordinate ([0, 1] for Q).
    """
    _require_mdtraj()

    try:
        # Process trajectory input
        if isinstance(traj, str):
            if native_file is None:
                traj_obj = md.load(traj)
            else:
                traj_obj = md.load(traj, top=native_file)
        elif isinstance(traj, md.Trajectory):
            traj_obj = traj  # already in proper format
        elif isinstance(traj, np.ndarray):
            traj_obj = md.Trajectory(xyz=traj, topology=md.load(native_file).topology)
        else:
            raise TypeError("Unsupported type for traj. Must be str, md.Trajectory, or np.ndarray.")

        # Cached native-contact definition (depends only on the native file).
        native_contacts, r0 = get_native_contacts(native_file)

        # Slice frames if requested
        frames = traj_obj[frame_range[0]:frame_range[1]] if frame_range is not None else traj_obj

        # Calculate Q
        r = md.compute_distances(frames, native_contacts)
        q = compute_q(r, r0)
        d_range = np.array([0, 1])
        return q, d_range

    except Exception as e:
        # Single, well-defined fallback: retry once with the initial structure.
        # Guard against infinite recursion and against an undefined fallback so
        # that genuine errors surface instead of being silently masked.
        if init_file is None or (isinstance(traj, str) and traj == init_file):
            raise
        warnings.warn(
            f"best_hummer_q failed for {traj!r} ({e}); falling back to init_file."
        )
        return best_hummer_q(init_file, native_file=native_file, frame_range=frame_range)


#############################################################################################################

# Cache of reference structures / target RMSD ranges keyed by file path, to
# avoid reloading the same PDB/GRO on every walker every cycle.
_RMSD_REF_CACHE = {}


def _load_rmsd_reference(top_file, unfolded_file):
    """Return (and cache) (ref_traj, ca_indices, target_rmsd) for backbone RMSD."""
    _require_mdtraj()

    key = (top_file, unfolded_file)
    cached = _RMSD_REF_CACHE.get(key)
    if cached is not None:
        return cached

    ref = md.load(top_file)
    atom_indices = ref.topology.select("name CA")

    tar = md.load(unfolded_file)
    tar.superpose(ref, atom_indices=atom_indices)
    tar_rmsd = md.rmsd(tar, ref, atom_indices=atom_indices)[0]

    result = (ref, atom_indices, tar_rmsd)
    _RMSD_REF_CACHE[key] = result
    return result


def RMSD_Backbone(traj, top_file=None, init_file=None, unfolded_file=None):
    """
    Compute backbone RMSD (CA atoms) using MDTraj.

    Parameters
    ----------
    traj : str, np.ndarray, or md.Trajectory
        Trajectory file path, coordinates array, or preloaded trajectory.
    top_file : str
        Topology / reference structure file (PDB, etc.).
    unfolded_file : str
        Structure used to define the RMSD range [0, target RMSD].

    Returns
    -------
    rmsds : np.ndarray
        RMSD values for each frame (aligned on CA atoms).
    drange : np.ndarray
        Array [0, target RMSD].
    """
    _require_mdtraj()

    try:
        if unfolded_file is None:
            raise ValueError("unfolded_file is required for determining rmsd range.")

        # Load or convert trajectory
        if isinstance(traj, str):
            if top_file is None:
                traj_obj = md.load(traj)
            else:
                traj_obj = md.load(traj, top=top_file)
        elif isinstance(traj, md.Trajectory):
            traj_obj = traj  # already in correct format
        elif isinstance(traj, np.ndarray):
            if top_file is None:
                raise ValueError("top_file required for NumPy coordinate array.")
            top = md.load(top_file).topology
            traj_obj = md.Trajectory(xyz=traj, topology=top)
        else:
            raise TypeError("traj must be str, md.Trajectory, or np.ndarray.")

        # Cached reference structure, CA indices and target RMSD range.
        ref, atom_indices, tar_rmsd = _load_rmsd_reference(top_file, unfolded_file)

        # Align trajectory to reference on CA atoms and compute RMSD.
        traj_obj.superpose(ref, atom_indices=atom_indices)
        rmsds = md.rmsd(traj_obj, ref, atom_indices=atom_indices)

        drange = np.array([0, tar_rmsd])
        return rmsds, drange

    except Exception as e:
        # Single, well-defined fallback (see best_hummer_q for rationale).
        if init_file is None or (isinstance(traj, str) and traj == init_file):
            raise
        warnings.warn(
            f"RMSD_Backbone failed for {traj!r} ({e}); falling back to init_file."
        )
        return RMSD_Backbone(init_file, top_file=top_file, init_file=init_file,
                             unfolded_file=unfolded_file)
