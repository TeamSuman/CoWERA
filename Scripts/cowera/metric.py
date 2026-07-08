import numpy as np
from joblib import Parallel, delayed

from cowera.features import best_hummer_q, RMSD_Backbone

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


def scale_weights(weights, increment):
    """Min-max scale the per-walker initial intensity I0 (function of projection).

    Returns 0.5 for all walkers when the range is degenerate so that the
    resulting intensities stay finite and uniform.
    """
    weights = np.asarray(weights, dtype=float)
    wmin = weights.min()
    wmax = weights.max()
    if wmax - wmin < 1e-12:
        scaled = np.full_like(weights, 0.5)
    else:
        scaled = (weights - wmin) / (wmax - wmin)
    return np.where(increment == 1, scaled, 1.0 - scaled)


def compute_intensity(weights_scaled, phases_scaled):
    """Combine scaled I0 and scaled phase into normalized intensity (Eq. 4/C4).

    ``interference = I0_scaled * sqrt(1 + phase_scaled)`` then normalized by its
    max. Guards against an all-zero interference vector.
    """
    interference = weights_scaled * np.sqrt(1.0 + phases_scaled)
    imax = interference.max()
    if imax <= 0.0:
        return np.ones_like(interference) / max(len(interference), 1)
    return interference / imax


def relevant_change_points(changes, n_d):
    """Build the segment boundaries used for the phase calculation (Appendix B).

    ``changes`` are the changepoint indices returned by the detector. We keep a
    short pre-transition buffer of ``n_d`` before the most recent changepoint
    when that is well-defined, otherwise fall back to the raw changepoints.
    """
    changes = np.asarray(changes)
    # NOTE: precedence fix -- the intent is "all interior changepoints, shifted
    # back by n_d, are still positive", i.e. np.all((changes[:-1] - n_d) > 0).
    if changes.size >= 1 and np.all((changes[:-1] - n_d) > 0):
        return np.concatenate(([0], changes[:-1] - n_d, changes[-1:]))
    return np.concatenate(([0], changes))


# --------------------------------------------------------------------------- #
# Distance / projection calculator
# --------------------------------------------------------------------------- #

class Calculate_Distances:
    def __init__(self, feat, increment, native_file=None, init_file=None, tar_file=None,
                 top_file=None, distance_criterion="pairwise_rmsd"):
        super().__init__()
        self._feat = feat
        self.native_file = native_file
        self.init_file = init_file
        self.tar_file = tar_file
        self.top_file = top_file
        self.increment = increment
        self.distance_criterion = distance_criterion

        # Cache for the MDAnalysis reference topology used by the pairwise-RMSD
        # criterion, built once and reused (previously two Universes were
        # constructed for *every* walker pair, every cycle).
        self._mda_ref = None
        self._mda_backbone = None

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

    def pairwise_distance_matrix(self, walkers):
        """Compute the full symmetric pairwise distance matrix for all walkers.

        Vectorized for the ``pairwise_rmsd`` criterion: the backbone atom
        selection is computed once and reused for every pair, avoiding the
        O(N^2) MDAnalysis Universe construction in the previous implementation.
        """
        n = len(walkers)
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

    def project_new_frames(self, i, path, offset):
        """Project only the frames produced since ``offset`` for walker ``i``.

        Returns ``(new_projection, drange, total_frames, was_reset)``. A reset
        (the trajectory now has fewer frames than were previously consumed, e.g.
        after a warp recreated the file) is reported so the caller can clear the
        stale in-memory history. This is the seam that Phase 3 will replace with
        CV values produced inline by the runner, removing the disk read entirely.
        """
        from cowera.cv_history import incremental_window

        full, drange = self._load_projection(i, path)
        n_frames = len(full)
        was_reset, start, total = incremental_window(offset, n_frames)
        return np.asarray(full[start:]), drange, total, was_reset

    def phase_from_projection(self, projection, drange, n_d, n_bins=100):
        """Compute ``(bins, phase, weight)`` from an in-memory CV series.

        Pure analysis (no disk I/O). Implements the relevant-history /
        phase calculation of Appendix B/C with NaN-safe helpers.
        """
        import ruptures as rpt

        projection = np.asarray(projection, dtype=float)

        if len(projection) <= 1:
            # Walker was warped this cycle (no meaningful history).
            return np.zeros(n_d), 0.0, projection[0] if len(projection) else 0.0

        drange = np.asarray(drange)[::self.increment]

        # Fast changepoints over the (in-memory) accumulated history.
        algo = rpt.KernelCPD(kernel="linear", min_size=2).fit(projection)
        changes = np.array(algo.predict(pen=0.01))
        change_points = relevant_change_points(changes, n_d)

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

        weights_scaled = scale_weights(weights, self.increment)
        phases_scaled = scale_phases(phases)

        fraction_unique = np.mean([
            len(np.unique(bins_arr[i])) / bins_arr.shape[1]
            for i in range(bins_arr.shape[0])
        ])
        if fraction_unique < 0.2:
            n_bins = min(max_bins, int((n_bins - 1) * bin_increase_factor))
        elif fraction_unique > 0.8:
            n_bins = max(10, int((n_bins - 1) * bin_decrease_factor))

        intensity = compute_intensity(weights_scaled, phases_scaled)
        return intensity, n_bins

    def intensity_from_projections(self, projections, dranges, n_d, it,
                                   n_bins=100, max_bins=125, n_jobs=-1):
        """Compute intensities directly from in-memory CV histories.

        This is the disk-free analysis path used by the resampler: it consumes
        the accumulated per-walker projection arrays maintained by
        :class:`cowera.cv_history.CVHistory` instead of re-reading DCD files.
        """
        n_walkers = len(projections)
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.phase_from_projection)(projections[i], dranges[i], n_d, n_bins)
            for i in range(n_walkers)
        )
        bins_list = [r[0] for r in results]
        phase_arr = [r[1] for r in results]
        weight_arr = [r[2] for r in results]
        return self.intensity_from_phases(bins_list, phase_arr, weight_arr,
                                           n_walkers, it, n_bins, max_bins)

    def intensity_calculation(self, n_walkers, path, n_d, it,
                              n_bins=100, max_bins=125,
                              bin_increase_factor=1.2, bin_decrease_factor=0.8,
                              n_jobs=-1):
        """Disk-based compatibility wrapper (kept for the legacy code path)."""
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.phase_calculation)(i, path, n_d, n_bins)
            for i in range(n_walkers)
        )
        bins_list = [r[0] for r in results]
        phase_arr = [r[1] for r in results]
        weight_arr = [r[2] for r in results]
        return self.intensity_from_phases(bins_list, phase_arr, weight_arr,
                                           n_walkers, it, n_bins, max_bins,
                                           bin_increase_factor, bin_decrease_factor)
