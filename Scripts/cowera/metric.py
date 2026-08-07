import logging
import os

import numpy as np
from joblib import Parallel, delayed

import cowera.features as _features
from cowera.features import best_hummer_q, RMSD_Backbone


def _resolve_n_jobs(n_jobs):
    """Choose a joblib worker count that respects the scheduler CPU allocation.

    ``joblib``'s ``n_jobs=-1`` grabs every logical core, oversubscribing a
    SLURM/PBS cgroup allocation and contending with the per-walker MD processes.
    When ``n_jobs`` is left at the auto default (None or -1) we instead honor
    ``SLURM_CPUS_PER_TASK`` / ``PBS_NP`` if present, falling back to -1 only
    outside a scheduler.
    """
    if n_jobs is not None and n_jobs != -1:
        return n_jobs
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NP", "OMP_NUM_THREADS"):
        val = os.environ.get(var)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    return -1


def euclidean_matrix_from_projections(projections):
    """Vectorized pairwise Euclidean distance matrix of per-walker projections.

    Exact, O(N^2)-scalar replacement for building the matrix by calling
    ``image_distance`` for every pair -- i.e. ``norm(get_proj_coord(s_i) -
    get_proj_coord(s_j))``, which re-projected each state ~2(N-1) times per cycle
    (~N^2 expensive RMSD/Q evaluations when only N are needed). Reusing the
    already-computed projections gives numerically identical distances with N (not
    ~N^2) projections. ``projections`` is a sequence of per-walker projection
    arrays (or scalars); the result is the symmetric (N, N) distance matrix.
    """
    P = np.asarray(projections, dtype=float)
    if P.ndim == 1:                      # scalar projection per walker
        return np.abs(P[:, None] - P[None, :])
    P = P.reshape(P.shape[0], -1)        # (N, d) — norm over the CV dimensions
    return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)

# Heavy / optional dependencies are imported lazily inside the methods that use
# them (ruptures for changepoint detection, MDAnalysis for the pairwise RMSD
# distance criterion) so that the pure-numpy algorithm cores in this module stay
# importable and unit-testable without the full MD stack.


# --------------------------------------------------------------------------- #
# Pure-numpy algorithm cores (unit-testable without mdtraj/MDAnalysis/ruptures)
# --------------------------------------------------------------------------- #

def phase_from_bins(bins_segment):
    """Mean signed bin displacement over a segment (paper Eq. 3 / C2).

    NaN-safe: a segment with fewer than two points has no transitions and is
    treated as fully stalled (phase 0.0).
    """
    seg = np.asarray(bins_segment)
    if seg.size < 2:
        return 0.0
    return float(np.sign(np.diff(seg)).mean())


def scale_phases(phases):
    """Scale phases to [-1, 1] across the ensemble (paper Eq. C3).

    Returns zeros when all phases are equal (degenerate range), avoiding the
    divide-by-zero that previously produced NaNs that propagated into the
    intensity and corrupted resampling.
    """
    phases = np.asarray(phases, dtype=float)
    pmin = np.min(phases)
    pmax = np.max(phases)
    if pmax - pmin < 1e-12:
        return np.zeros_like(phases)
    return 2.0 * (phases - pmin) / (pmax - pmin) - 1.0


def scale_weights(weights, increment, i0_mode="projection"):
    """Compute the per-walker initial intensity I0 (paper Appendix C).

    ``i0_mode="projection"`` (default): min-max scale the projection (0.5 for all
    walkers when the range is degenerate), directed by ``increment``.
    ``i0_mode="uniform"``: I0 = 1 for every walker (the paper's "uniformly one"
    option). Any other value raises.
    """
    weights = np.asarray(weights, dtype=float)
    if i0_mode == "uniform":
        return np.ones_like(weights)
    if i0_mode != "projection":
        raise ValueError(f"Unrecognized i0_mode: {i0_mode!r} (use 'projection' or 'uniform')")
    wmin = weights.min()
    wmax = weights.max()
    if wmax - wmin < 1e-12:
        scaled = np.full_like(weights, 0.5)
    else:
        scaled = (weights - wmin) / (wmax - wmin)
    return np.where(increment == 1, scaled, 1.0 - scaled)


def compute_intensity(weights_scaled, phases_scaled, use_phase=True):
    """Combine scaled I0 and scaled phase into normalized intensity (Eq. 4/C4).

    ``interference = I0_scaled * sqrt(1 + phase_scaled)`` then normalized by its
    max. Guards against an all-zero interference vector.

    ``use_phase=False`` drops the coherence phase entirely (``interference = I0``),
    reducing CoWERA to the proximity-only *targeted-WE baseline* of the paper's
    comparison (resampling guided solely by I0, i.e. distance to the target).
    """
    if use_phase:
        interference = weights_scaled * np.sqrt(1.0 + phases_scaled)
    else:
        interference = np.asarray(weights_scaled, dtype=float)
    imax = interference.max()
    if imax <= 0.0:
        return np.ones_like(interference) / max(len(interference), 1)
    return interference / imax


def relevant_change_points(changes, n_d, mode="published"):
    """Build the segment boundaries used for the phase calculation (Appendix B).

    ``changes`` are the changepoint indices returned by the detector. A short
    pre-transition buffer of ``n_d`` frames is kept before the most recent
    changepoint, so the phase is measured over the transition *plus* its run-up.

    ``mode`` selects the guard on that buffer -- these are NOT equivalent, and the
    choice changes the phase window, hence the resampling:

    "published" (DEFAULT) -- reproduces the released CoWERA code and therefore the
        manuscript numbers. The guard is ``np.all(changes[:-1] - n_d) > 0``, which
        evaluates as ``(all elements nonzero) > 0`` and is therefore True unless an
        interior changepoint lands exactly on ``n_d``: the buffer is applied almost
        always. NB this can make ``changes[-2] - n_d`` NEGATIVE, which numpy slicing
        silently reads as "the last |x| frames" -- a quirk of the original, retained
        here deliberately so results remain comparable to the published values.

    "strict" -- applies the buffer only when every interior changepoint exceeds
        ``n_d`` (``np.all((changes[:-1] - n_d) > 0)``), avoiding the negative-index
        wraparound. This is the arguably-intended reading, but it yields a SHORTER
        phase window whenever an interior changepoint <= n_d (common right after a
        warp reset, when the history is short), so it is a real behavioural change
        and must not be used for reproduction work.
    """
    changes = np.asarray(changes)
    if mode == "strict":
        keep = changes.size >= 1 and np.all((changes[:-1] - n_d) > 0)
    else:
        keep = np.all(changes[:-1] - n_d) > 0
    if keep:
        return np.concatenate(([0], changes[:-1] - n_d, changes[-1:]))
    return np.concatenate(([0], changes))


# --------------------------------------------------------------------------- #
# Distance / projection calculator
# --------------------------------------------------------------------------- #

class Calculate_Distances:
    def __init__(self, feat, increment, native_file=None, init_file=None, tar_file=None,
                 top_file=None, distance_criterion="pairwise_rmsd",
                 i0_mode="projection", use_phase=True, changepoint_window=None,
                 incremental_cv_read=True, verify_incremental_cv=False,
                 changepoint_algo="kernelcpd", changepoint_penalty=0.01,
                 relevant_history_mode="published"):
        super().__init__()
        self._feat = feat
        self.native_file = native_file
        self.init_file = init_file
        self.tar_file = tar_file
        self.top_file = top_file
        self.increment = increment
        self.distance_criterion = distance_criterion
        # I0 form (Appendix C) and whether the coherence phase is used. use_phase
        # False + i0_mode 'projection' reproduces the paper's targeted-WE baseline.
        self.i0_mode = i0_mode
        self.use_phase = use_phase
        # P3b: cap the CV history the changepoint detector sees to the most recent
        # `changepoint_window` frames -> KernelCPD is O(W^2) per walker instead of
        # O(T^2) growing with the run. None = unbounded (exact, the default). Only
        # the most recent changepoint (the current phase boundary) is used, so a
        # window >= the relevant-history length reproduces the unbounded result.
        self.changepoint_window = changepoint_window

        # A2: read ONLY the new frames from each walker's DCD each cycle instead of
        # md.load()ing the whole growing trajectory (was O(T), the dominant growing
        # cost -- ~19% and climbing). `incremental_cv_read` enables it;
        # `verify_incremental_cv` additionally recomputes the full-load projection
        # and asserts they are identical (validation; slower). Default OFF = the exact
        # full-load path, until validated on a host (this repo cannot import mdtraj).
        self.incremental_cv_read = incremental_cv_read
        self.verify_incremental_cv = verify_incremental_cv
        self._md_top = None                 # cached mdtraj topology (lazily built)

        # D-2 (docs/MANUSCRIPT_CONSISTENCY.md): the manuscript (Appendix B) specifies
        # PELT with a linear kernel; the code historically hardcoded a *different*
        # search, KernelCPD(kernel="linear"), with an *undocumented* penalty 0.01.
        # Both are now selectable so the published setup can be reproduced/compared.
        #   "kernelcpd" -> ruptures.KernelCPD(kernel="linear")   [historical default]
        #   "pelt"      -> ruptures.Pelt(model="l2")             [paper Appendix B]
        # (KernelCPD's linear kernel and PELT's l2 cost are the same cost; the
        # difference is the search algorithm.) The penalty stays undocumented in the
        # paper -- keep it configurable rather than silently baked in.
        self.changepoint_algo = changepoint_algo
        self.changepoint_penalty = changepoint_penalty
        # Which guard to use for the n_d pre-transition buffer in the phase window.
        # "published" reproduces the released code (and the manuscript numbers);
        # "strict" is the arguably-intended reading but shortens the window whenever
        # an interior changepoint <= n_d. See relevant_change_points().
        self.relevant_history_mode = relevant_history_mode

        # Cache for the MDAnalysis reference topology used by the pairwise-RMSD
        # criterion, built once and reused (previously two Universes were
        # constructed for *every* walker pair, every cycle).
        self._mda_ref = None
        self._mda_backbone = None

    def __getstate__(self):
        # The cached MDAnalysis Universe (pairwise_rmsd path) is NOT deep-copyable
        # (its __deepcopy__ signature errors) nor cleanly picklable, and it is only a
        # lazily-rebuilt cache. Exclude it so the end-of-run component snapshot and
        # pkl checkpoints work on pairwise_rmsd systems (e.g. Trp-cage); it is
        # re-created on next use via _get_mda_template().
        state = self.__dict__.copy()
        state["_mda_ref"] = None
        state["_mda_backbone"] = None
        state["_md_top"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ----------------------- pairwise RMSD (MDAnalysis) -------------------- #

    def _get_mda_template(self):
        """Lazily build and cache the MDAnalysis Universe template + backbone sel."""
        if self._mda_ref is None:
            import MDAnalysis as mda
            self._mda_ref = mda.Universe(self.native_file)
            self._mda_backbone = self._mda_ref.select_atoms("backbone").indices
        return self._mda_ref, self._mda_backbone

    def image_distance(self, state1, state2):
        if self.distance_criterion == "pairwise_rmsd":
            try:
                from MDAnalysis.analysis.rms import rmsd

                coord1 = np.asarray(state1['positions'])
                coord2 = np.asarray(state2['positions'])

                _, backbone = self._get_mda_template()

                # backbone atom selection applied directly to the coordinate
                # arrays -- no per-pair Universe construction.
                return rmsd(coord1[backbone], coord2[backbone],
                            center=True, superposition=True)

            except Exception as e:
                print(f"RMSD computation failed: {e}")
                return None

        elif self.distance_criterion == "euclidean":
            try:
                image1 = self.get_proj_coord(state1)
                image2 = self.get_proj_coord(state2)
                return np.linalg.norm(image1 - image2)
            except Exception as e:
                print(f"Euclidean distance computation failed: {e}")
                return None
        else:
            raise ValueError(
                f"Unrecognized distance criterion: {self.distance_criterion} "
                'use "pairwise_rmsd" or "euclidean"'
            )

    def pairwise_distance_matrix(self, walkers, projections=None):
        """Compute the full symmetric pairwise distance matrix for all walkers.

        For the ``euclidean`` criterion the distance is the norm between per-walker
        projections; reusing the already-computed ``projections`` (or projecting
        once here) makes this an O(N^2)-scalar operation instead of ~N^2 expensive
        re-projections inside ``image_distance`` -- numerically identical.

        For ``pairwise_rmsd`` the backbone selection is computed once and reused
        for every pair, avoiding per-pair MDAnalysis Universe construction.
        """
        n = len(walkers)

        if self.distance_criterion == "euclidean":
            if projections is None:
                projections = [self.get_proj_coord(w.state) for w in walkers]
            return euclidean_matrix_from_projections(projections)

        dist_mat = np.zeros((n, n))
        if self.distance_criterion == "pairwise_rmsd":
            from MDAnalysis.analysis.rms import rmsd

            _, backbone = self._get_mda_template()
            coords = [np.asarray(w.state['positions'])[backbone] for w in walkers]

            for i in range(n):
                for j in range(i + 1, n):
                    d = rmsd(coords[i], coords[j], center=True, superposition=True)
                    dist_mat[i, j] = dist_mat[j, i] = d
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    d = self.image_distance(walkers[i].state, walkers[j].state)
                    dist_mat[i, j] = dist_mat[j, i] = d

        return dist_mat

    # ----------------------------- projection ----------------------------- #

    def get_proj_coord(self, state):
        """
        Compute the 'image' of a walker state — e.g., projecting to a collective variable.
        """
        if self._feat == "best_hummer_q":
            q, _ = best_hummer_q(
                traj=state['positions'],
                native_file=self.native_file,
            )
            proj_coord = q

        elif self._feat == "rmsd_backbone":
            if self.increment == 1:
                rmsd_val, _ = RMSD_Backbone(
                    traj=state['positions'],
                    top_file=self.top_file,
                    init_file=self.init_file,
                    unfolded_file=self.tar_file
                )
            else:
                rmsd_val, _ = RMSD_Backbone(
                    traj=state['positions'],
                    top_file=self.top_file,
                    init_file=self.init_file,
                    unfolded_file=self.init_file
                )
            proj_coord = rmsd_val
        else:
            raise ValueError(f"Unrecognized feature selection: {self._feat}")
        return proj_coord

    def get_bin_edges(self, x_min=0, x_max=1, n_bins=100):
        return np.linspace(x_min, x_max, n_bins + 1)

    def get_bin(self, x, bin_edges):
        return np.digitize(x, bin_edges)

    # --------------------------- phase per walker ------------------------- #

    def _load_projection(self, i, path, frame_slice=None):
        """Load a walker's trajectory and project it onto the progress coordinate.

        Returns ``(projection, drange)``. If ``frame_slice`` is given it is
        applied to the trajectory before projection so that only the requested
        (e.g. newly produced) frames are projected -- avoiding re-projection of
        the entire accumulated history every cycle.
        """
        if self._feat == "best_hummer_q":
            traj = f"{path}walker_{i}.dcd"
            if frame_slice is not None:
                import mdtraj as md
                traj = md.load(traj, top=self.native_file)[frame_slice]
            projection, drange = best_hummer_q(
                traj=traj,
                native_file=self.native_file,
                init_file=self.init_file,
            )
        elif self._feat == "rmsd_backbone":
            traj = f"{path}walker_{i}.dcd"
            if frame_slice is not None:
                import mdtraj as md
                traj = md.load(traj, top=self.top_file)[frame_slice]
            unfolded = self.tar_file if self.increment == 1 else self.init_file
            projection, drange = RMSD_Backbone(
                traj=traj,
                init_file=self.init_file,
                unfolded_file=unfolded,
                top_file=self.top_file,
            )
        else:
            raise ValueError(f"Unrecognized feature selection: {self._feat}")
        return projection, drange

    def _load_walker_traj(self, i, path):
        """Load walker ``i``'s DCD as an mdtraj Trajectory, with the same
        init-structure fallback (and fallback counter) as the feature functions.
        """
        import mdtraj as md

        traj_file = f"{path}walker_{i}.dcd"
        top = self.native_file if self._feat == "best_hummer_q" else self.top_file
        try:
            return md.load(traj_file, top=top)
        except Exception as exc:
            if self.init_file is None:
                raise
            _features._FALLBACK_COUNT += 1
            logging.warning(
                "Could not load %r (%s); falling back to init_file "
                "(the initial-structure CV is substituted for this walker).",
                traj_file, exc)
            return md.load(self.init_file, top=top)

    def _project_traj(self, traj_obj):
        """Project an already-loaded mdtraj Trajectory onto the progress coordinate.

        No init-structure fallback here (the trajectory is already in hand); a
        failure is a genuine error. Returns ``(projection, drange)``.
        """
        if self._feat == "best_hummer_q":
            return best_hummer_q(traj=traj_obj, native_file=self.native_file, init_file=None)
        elif self._feat == "rmsd_backbone":
            unfolded = self.tar_file if self.increment == 1 else self.init_file
            return RMSD_Backbone(traj=traj_obj, init_file=None,
                                 unfolded_file=unfolded, top_file=self.top_file)
        raise ValueError(f"Unrecognized feature selection: {self._feat}")

    def _md_topology(self):
        """Cached mdtraj topology for building Trajectories from raw DCD frames."""
        if self._md_top is None:
            import mdtraj as md
            top = self.native_file if self._feat == "best_hummer_q" else self.top_file
            self._md_top = md.load_topology(top)
        return self._md_top

    def _project_new_full(self, i, path, offset):
        """Exact reference path: md.load the WHOLE DCD, slice to the new frames, and
        project. O(total-frames) I/O every cycle -- the A2 target -- but always
        correct. Returns ``(new_projection, drange, total_frames, was_reset)``."""
        from cowera.cv_history import incremental_window
        traj_obj = self._load_walker_traj(i, path)
        n_frames = traj_obj.n_frames
        was_reset, start, total = incremental_window(offset, n_frames)
        proj, drange = self._project_traj(traj_obj[start:])
        return np.asarray(proj), drange, total, was_reset

    def _project_new_incremental(self, i, path, offset):
        """A2: read ONLY frames ``[offset:]`` from the DCD via random access and
        project them -- O(new frames) instead of O(total). The per-frame CV is
        frame-independent, so this is numerically identical to ``_project_new_full``.
        DCD positions are angstroms; mdtraj Trajectories are nm (hence /10)."""
        from cowera.cv_history import incremental_window
        import mdtraj as md
        traj_file = f"{path}walker_{i}.dcd"
        with md.formats.DCDTrajectoryFile(traj_file) as f:
            n_frames = len(f)
            was_reset, start, total = incremental_window(offset, n_frames)
            f.seek(start)
            xyz, _cell_lengths, _cell_angles = f.read()   # frames from `start` onward
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.shape[0] == 0:                              # no new frames this cycle
            return np.asarray([]), np.asarray([0.0, 0.0]), total, was_reset
        traj_obj = md.Trajectory(xyz=xyz / 10.0, topology=self._md_topology())
        proj, drange = self._project_traj(traj_obj)
        return np.asarray(proj), drange, total, was_reset

    def project_new_frames(self, i, path, offset):
        """Project only the frames produced since ``offset`` for walker ``i``.

        Returns ``(new_projection, drange, total_frames, was_reset)``. A reset (the
        file now has fewer frames than were consumed, e.g. a warp recreated it) is
        reported so the caller can clear the stale in-memory history.

        A2: when ``incremental_cv_read`` is set, reads only the NEW frames from the
        DCD (O(new) instead of O(total) -- the dominant growing cost). DEFENSIVE: on
        any error it falls back to the exact full load, and with
        ``verify_incremental_cv`` it recomputes the full-load projection and asserts
        they match -- so correctness is never at risk. Default: the full-load path.
        """
        if not self.incremental_cv_read:
            return self._project_new_full(i, path, offset)
        try:
            result = self._project_new_incremental(i, path, offset)
        except Exception as exc:
            logging.warning("A2 incremental CV read failed for walker %s (%s); "
                            "falling back to full DCD load.", i, exc)
            return self._project_new_full(i, path, offset)
        if self.verify_incremental_cv:
            ref = self._project_new_full(i, path, offset)
            a = np.asarray(result[0], dtype=float)
            b = np.asarray(ref[0], dtype=float)
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-6, rtol=0.0):
                logging.error("A2 VERIFY MISMATCH walker %s: incremental != full load "
                              "(max|Δ| %s, shapes %s vs %s); using full load.", i,
                              (np.abs(a - b).max() if a.shape == b.shape else "n/a"),
                              a.shape, b.shape)
                return ref
        return result

    def _detect_changes(self, projection):
        """Changepoint indices for ``projection`` (always ends with its length).

        ``changepoint_algo`` picks the ruptures search (see __init__ / D-2):
          "kernelcpd" -> KernelCPD(kernel="linear")  [historical default]
          "pelt"      -> Pelt(model="l2")            [paper Appendix B]
        both at ``changepoint_penalty`` (default 0.01, undocumented in the paper).

        Guards short/failing signals: min_size=2 needs >= 2*min_size samples or
        ruptures raises BadSegmentationParameters (which would crash the first cycles
        of a run with few frames/segment); on a too-short signal or any detector
        failure, degrade to a single segment (the whole history).
        """
        _MIN_SIZE = 2
        n = len(projection)
        if n < 2 * _MIN_SIZE:
            return np.array([n])
        try:
            import ruptures as rpt
            if self.changepoint_algo == "pelt":
                algo = rpt.Pelt(model="l2", min_size=_MIN_SIZE).fit(projection)
            else:
                algo = rpt.KernelCPD(kernel="linear", min_size=_MIN_SIZE).fit(projection)
            return np.array(algo.predict(pen=self.changepoint_penalty))
        except Exception:
            return np.array([n])

    def phase_from_projection(self, projection, drange, n_d, n_bins=100):
        """Compute ``(bins, phase, weight)`` from an in-memory CV series.

        Pure analysis (no disk I/O). Implements the relevant-history /
        phase calculation of Appendix B/C with NaN-safe helpers.
        """
        projection = np.asarray(projection, dtype=float)

        if len(projection) <= 1:
            # Walker was warped this cycle (no meaningful history).
            return np.zeros(n_d), 0.0, projection[0] if len(projection) else 0.0

        drange = np.asarray(drange)[::self.increment]

        # P3b: find the most recent changepoint (the current phase boundary, Appendix
        # B) with a GROWING window so KernelCPD runs on O(relevant-history) frames,
        # not O(T) growing with the run (the dominant CPU cost at scale). EXACT by
        # construction: expand the window until its last changepoint is INTERIOR (the
        # whole current segment is captured) or the full history is used -- so it
        # reproduces the unbounded result while bounding typical cost to O(W^2) with
        # W = relevant-history length << T. `changepoint_window` seeds the initial
        # window; None -> straight to the full history (unbounded, exact default).
        # `weight = projection[-1]` (the current CV value) is unaffected either way.
        T = len(projection)
        seed = self.changepoint_window
        if seed is None or seed >= T:
            changes = self._detect_changes(projection)
        else:
            W = max(int(seed), 4)                      # >= 2*min_size
            prev_seg = None                            # last-changepoint position (frames from end)
            while True:
                sub = projection[-W:]
                changes = self._detect_changes(sub)
                seg = (len(sub) - int(changes[-2])) if len(changes) >= 2 else None
                # Accept only when the last changepoint is STABLE across a window
                # doubling (KernelCPD's penalty is global, so a single small window can
                # place a spurious changepoint); else keep expanding, up to the full
                # history. This makes the windowed result match the unbounded one.
                if W >= T or (seg is not None and seg == prev_seg):
                    projection = sub                   # analyze on the window (holds last segment)
                    break
                prev_seg = seg
                W = min(W * 2, T)

        change_points = relevant_change_points(changes, n_d,
                                               mode=self.relevant_history_mode)

        weight = projection[-1]

        bin_edges = self.get_bin_edges(x_min=drange[0], x_max=drange[1], n_bins=n_bins)
        current_bins = np.digitize(projection, bin_edges, right=True)

        seg = current_bins[change_points[-2]:change_points[-1]]
        phase = phase_from_bins(seg)

        return current_bins[-n_d:], phase, weight

    def phase_calculation(self, i, path, n_d, n_bins=100):
        """Disk-based compatibility wrapper: load the trajectory then analyze."""
        projection, drange = self._load_projection(i, path)
        if len(projection) == 1:
            return np.zeros(n_d), 0.0, projection[0]
        return self.phase_from_projection(projection, drange, n_d, n_bins)

    # ----------------------------- intensity ------------------------------ #

    def intensity_from_phases(self, bins_list, phase_arr, weight_arr, n_walkers, it,
                              n_bins=100, max_bins=125,
                              bin_increase_factor=1.2, bin_decrease_factor=0.8):
        """Combine per-walker (bins, phase, weight) into normalized intensities.

        Pure-numpy (no disk / changepoint); shared by the in-memory and
        disk-based code paths. Also performs the adaptive bin adjustment
        (Appendix F) and returns the (possibly) updated ``n_bins``.
        """
        bins_arr = np.array(bins_list, dtype=float)

        if (bins_arr == 0).all():
            print(f"Cycle {it}: no valid walkers, uniform weights.")
            return np.ones(n_walkers) / n_walkers, n_bins

        phases = np.array(phase_arr, dtype=float)
        weights = np.array(weight_arr, dtype=float)

        weights_scaled = scale_weights(weights, self.increment, self.i0_mode)
        phases_scaled = scale_phases(phases)

        fraction_unique = np.mean([
            len(np.unique(bins_arr[i])) / bins_arr.shape[1]
            for i in range(bins_arr.shape[0])
        ])
        if fraction_unique < 0.2:
            n_bins = min(max_bins, int((n_bins - 1) * bin_increase_factor))
        elif fraction_unique > 0.8:
            n_bins = max(10, int((n_bins - 1) * bin_decrease_factor))

        intensity = compute_intensity(weights_scaled, phases_scaled, self.use_phase)
        return intensity, n_bins

    def intensity_from_projections(self, projections, dranges, n_d, it,
                                   n_bins=100, max_bins=125,
                                   bin_increase_factor=1.2, bin_decrease_factor=0.8,
                                   n_jobs=-1):
        """Compute intensities directly from in-memory CV histories.

        This is the disk-free analysis path used by the resampler: it consumes
        the accumulated per-walker projection arrays maintained by
        :class:`cowera.cv_history.CVHistory` instead of re-reading DCD files.

        ``bin_increase_factor`` / ``bin_decrease_factor`` are the adaptive-bin
        adjustment factors (paper Appendix F); previously this method silently
        dropped them, always using the ``intensity_from_phases`` defaults.
        """
        n_walkers = len(projections)
        results = Parallel(n_jobs=_resolve_n_jobs(n_jobs), prefer="threads")(
            delayed(self.phase_from_projection)(projections[i], dranges[i], n_d, n_bins)
            for i in range(n_walkers)
        )
        bins_list = [r[0] for r in results]
        phase_arr = [r[1] for r in results]
        weight_arr = [r[2] for r in results]
        return self.intensity_from_phases(bins_list, phase_arr, weight_arr,
                                           n_walkers, it, n_bins, max_bins,
                                           bin_increase_factor, bin_decrease_factor)

    def intensity_calculation(self, n_walkers, path, n_d, it,
                              n_bins=100, max_bins=125,
                              bin_increase_factor=1.2, bin_decrease_factor=0.8,
                              n_jobs=-1):
        """Disk-based compatibility wrapper (kept for the legacy code path)."""
        results = Parallel(n_jobs=_resolve_n_jobs(n_jobs), prefer="threads")(
            delayed(self.phase_calculation)(i, path, n_d, n_bins)
            for i in range(n_walkers)
        )
        bins_list = [r[0] for r in results]
        phase_arr = [r[1] for r in results]
        weight_arr = [r[2] for r in results]
        return self.intensity_from_phases(bins_list, phase_arr, weight_arr,
                                           n_walkers, it, n_bins, max_bins,
                                           bin_increase_factor, bin_decrease_factor)
