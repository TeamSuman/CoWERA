import random as rand

import logging
from eliot import start_action, log_call

import numpy as np
from wepy.resampling.resamplers.resampler import Resampler
from wepy.resampling.resamplers.clone_merge  import CloneMergeResampler
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from cowera.file_resampler import update_dcd_files
from cowera.features import get_fallback_count

class CoWERAResampler(CloneMergeResampler):
    r"""

    Resampler implementing the new REVO algorithm.

    """

    # fields for resampler data
    RESAMPLING_FIELDS = CloneMergeResampler.RESAMPLING_FIELDS
    RESAMPLING_SHAPES = CloneMergeResampler.RESAMPLING_SHAPES #+ (Ellipsis,)
    RESAMPLING_DTYPES = CloneMergeResampler.RESAMPLING_DTYPES #+ (np.int,)


    # fields that can be used for a table like representation
    RESAMPLING_RECORD_FIELDS = CloneMergeResampler.RESAMPLING_RECORD_FIELDS

    # fields for resampling data
    RESAMPLER_FIELDS = CloneMergeResampler.RESAMPLER_FIELDS + \
                       ('num_walkers', 'distance_array', 'variation', 'image_shape', 'images')
    RESAMPLER_SHAPES = CloneMergeResampler.RESAMPLER_SHAPES + \
                       ((1,), Ellipsis, (1,), Ellipsis, Ellipsis)
    RESAMPLER_DTYPES = CloneMergeResampler.RESAMPLER_DTYPES + \
                       (int, float, float, int, None)

    # fields that can be used for a table like representation
    RESAMPLER_RECORD_FIELDS = CloneMergeResampler.RESAMPLER_RECORD_FIELDS + \
                              ('variation',)


    def __init__(self,
                 distance=None,
                 run_id=None,
                 merge_dist=None,
                 pmax=0.25,
                 pmin=1e-12,
                 init_state=None,
                 seed=None,
                 info_file_path=None,
                 increment = None,
                 dcd_folder = None,
                 n_d = None,
                 mode = "greedy",
                 use_cv_history = True,
                 history_window = None,
                 bin_increase_factor = 1.2,
                 bin_decrease_factor = 0.8,
                 **kwargs):

        """Constructor for the REVO Resampler.

        Parameters
        ----------


        distance : object implementing Distance
            The distance metric to compare walkers.

        merge_dist : float
            The merge distance threshold. Units should be the same as
            the distance metric.

        init_state : WalkerState object
            Used for automatically determining the state image shape.

        seed : None or int, optional
            The random seed. If None, the system (random) one will be used.

        """

        # call the init methods in the CloneMergeResampler
        # superclass. We set the min and max number of walkers to be
        # constant
        super().__init__(pmin=pmin, pmax=pmax,
                         min_num_walkers=Ellipsis,
                         max_num_walkers=Ellipsis,
                         mode=mode,
                         **kwargs)

        assert distance is not None,  "Distance object must be given."
        assert init_state is not None,  "An  state must be given."

        # Directory

        self.info_file_path = info_file_path

        # the distance metric

        self.distance = distance

        # merge distance

        self.merge_dist = merge_dist

        # run index

        self.run_id = run_id

        # direction

        self.increment = increment

        # number of data per sement

        self.n_d = n_d

        # setting the random seed
        self.seed = seed

        self.mode = mode

        #dcd folder
        self.dcd_folder = dcd_folder

        # Unified random number generator. Previously three different RNGs
        # (``random``, ``numpy.random`` and the ``rand`` alias) were used while
        # only ``rand`` was seeded, which undermined run-to-run reproducibility.
        # We now drive every stochastic decision from a single seeded Generator.
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            rand.seed(seed)

        # Phase 2: in-memory CV history. When enabled, each walker's progress
        # coordinate time series is maintained in memory and updated
        # incrementally (only newly produced frames are projected each cycle)
        # instead of re-reading and re-projecting the entire accumulated DCD
        # every cycle. ``history_window`` optionally bounds the retained history
        # (and thus the changepoint cost). Set ``use_cv_history=False`` to fall
        # back to the legacy disk-based analysis path.
        self.use_cv_history = use_cv_history
        self.history_window = history_window
        self._cv_history = None

        # Adaptive-bin adjustment factors (paper Appendix F). Kept as attributes
        # so they can be set from config; forwarded to both analysis paths.
        self.bin_increase_factor = bin_increase_factor
        self.bin_decrease_factor = bin_decrease_factor

        # Most recent adaptive bin count returned by ``resample``. Snapshotted by
        # the pickle/checkpoint reporter so a restarted run resumes with the same
        # bin resolution instead of jumping back to the config's initial value.
        self._last_n_bins = None

        # Running total of feature-read fallbacks at the previous cycle, so we can
        # report the per-cycle delta (a nonzero value flags CV corruption risk).
        self._prev_fallback_count = 0

        # we do not know the shape and dtype of the images until
        # runtime so we determine them here
        #print("init state",init_state)
        image = self.distance.get_proj_coord(init_state)
        self.image_dtype = image.dtype

        #print(self.pmax)

    def resampler_field_dtypes(self):
        """ Finds out the datatype of the image.

        Returns
        -------
        datatypes : tuple of datatype
        The type of reasampler image.

        """

        # index of the image idx
        image_idx = self.resampler_field_names().index('images')

        # dtypes adding the image dtype
        dtypes = list(super().resampler_field_dtypes())
        dtypes[image_idx] = self.image_dtype

        return tuple(dtypes)


    def _calcvariation(self, num_walker_copies, distance_arr,walker_weights):

        # calculate  walker variation values

        walker_variations = distance_arr.copy() #* walker_weights
        variation = np.sum(distance_arr * num_walker_copies * walker_weights)

        return variation, walker_variations

    def decide(self, walker_weights, num_walker_copies, distance_arr, distance_matrix, images):
        """
        Optimize trajectory variation by resampling walkers efficiently.
        """
        num_walkers = len(walker_weights)

        variations = []
        merge_groups = [[] for _ in range(num_walkers)]
        walker_clone_nums = np.zeros(num_walkers, dtype=int)

        new_walker_weights = np.array(walker_weights)
        new_num_walker_copies = np.array(num_walker_copies)
        distance_matrix = np.array(distance_matrix)
        # Precompute merge eligibility matrix
        merge_mask = distance_matrix <= self.merge_dist

        variation, walker_variations = self._calcvariation(new_num_walker_copies, distance_arr, walker_weights)
        variations.append(variation)

        productive = True
        while productive:
            productive = False

            # Candidate for cloning
            max_candidates = [
                (walker_variations[i], i)
                for i in range(num_walkers)
                if new_num_walker_copies[i] >= 1 and
                (new_walker_weights[i] / (new_num_walker_copies[i] + 1) > self.pmin) and
                len(merge_groups[i]) == 0
            ]
            max_idx = max(max_candidates)[1] if max_candidates else None

            # Candidate for merging
            # min_candidates = [
            #     (images[i], i)
            #     for i in range(num_walkers)
            #     if new_num_walker_copies[i] == 1 and
            #     new_walker_weights[i] < self.pmax and
            #     merge_mask[i].any()
            # ]
            # min_idx = min(min_candidates)[1] if min_candidates else None

            min_candidates = [
                                (walker_variations[i], i)
                                for i in range(num_walkers)
                                if new_num_walker_copies[i] == 1
                                and (new_walker_weights[i] < self.pmax)
                            ]

            min_idx = min(min_candidates)[1] if min_candidates else None

            closewalk = None
            happen = False
            if min_idx is not None and max_idx is not None and min_idx != max_idx:
                # Vectorized candidate selection
                candidates = np.where(
                    (new_num_walker_copies == 1) &
                    (np.arange(num_walkers) != min_idx) &
                    (np.arange(num_walkers) != max_idx) &
                    ((new_walker_weights + new_walker_weights[min_idx]) < self.pmax)
                )[0]

                eligible = candidates[merge_mask[min_idx, candidates]]
                if len(eligible) > 0:
                    closewalk = self._rng.choice(eligible)

            #print(f"Max idx: {max_idx}, Max var: {walker_variations[max_idx] if max_idx is not None else 'N/A'}, Min idx: {min_idx}, Min var: {walker_variations[min_idx] if min_idx is not None else 'N/A'}, Close walk: {closewalk}, min dist : {distance_matrix[min_idx, closewalk] if closewalk is not None else 'N/A'}")
            if min_idx is not None and max_idx is not None and closewalk is not None:
                # r = np.random.uniform(0, new_walker_weights[closewalk] + new_walker_weights[min_idx])
                # if r < new_walker_weights[closewalk]:
                #     keep_idx, squash_idx = closewalk, min_idx
                # else:
                #     keep_idx, squash_idx = min_idx, closewalk
                keep_idx = closewalk
                squash_idx = min_idx

                new_num_walker_copies[squash_idx] = 0
                new_num_walker_copies[keep_idx] = 1
                new_num_walker_copies[max_idx] += 1

                new_variation, walker_variations = self._calcvariation(new_num_walker_copies, distance_arr, new_walker_weights)
                if self.mode == "greedy":
                    if new_variation > variation: # resampling is possible with these choices of walkers
                        variations.append(new_variation)

                        logging.info("Variance move to {} accepted".format(new_variation))

                        productive = True
                        variation = new_variation

                        # update weight
                        new_walker_weights[keep_idx] += new_walker_weights[squash_idx]
                        new_walker_weights[squash_idx] = 0.0

                        # add the squash index to the merge group
                        merge_groups[keep_idx].append(squash_idx)

                        # add the indices of the walkers that were already
                        # in the merge group that was just squashed
                        merge_groups[keep_idx].extend(merge_groups[squash_idx])

                        # reset the merge group that was just squashed to empty
                        merge_groups[squash_idx] = []

                        # increase the number of clones that the cloned
                        # walker has
                        walker_clone_nums[max_idx] += 1

                        logging.info("variance after selection: {}".format(new_variation))


                    # if not productive
                    else:
                        new_num_walker_copies[min_idx] = 1
                        new_num_walker_copies[closewalk] = 1
                        new_num_walker_copies[max_idx] -= 1


                elif self.mode == "probabilistic":
                    denom = new_variation + variation
                    prob = new_variation / denom if denom > 0 else 0.0
                    if self._rng.random() < prob:
                        variations.append(new_variation)
                        productive = True
                        variation = new_variation

                        new_walker_weights[keep_idx] += new_walker_weights[squash_idx]
                        new_walker_weights[squash_idx] = 0.0

                        merge_groups[keep_idx].append(squash_idx)
                        merge_groups[keep_idx].extend(merge_groups[squash_idx])
                        merge_groups[squash_idx] = []

                        walker_clone_nums[max_idx] += 1

                    else:
                        # revert changes if unproductive
                        new_num_walker_copies[min_idx] = 1
                        new_num_walker_copies[closewalk] = 1
                        new_num_walker_copies[max_idx] -= 1



        if (variations[-1] > variations[0]):
            happen = True
        else:
            happen = False

        # given we know what we want to clone to specific slots
        # (squashing other walkers) we need to determine where these
        # squashed walkers will be merged
        walker_actions = self.assign_clones(merge_groups, walker_clone_nums)

        # because there is only one step in resampling here we just
        # add another field for the step as 0 and add the walker index
        # to its record as well
        for walker_idx, walker_record in enumerate(walker_actions):
            walker_record['step_idx'] = np.array([0])
            walker_record['walker_idx'] = np.array([walker_idx])


        return walker_actions, variations[-1], happen

    def _update_cv_histories(self, walkers, folder, n_d):
        """Incrementally update the in-memory per-walker CV histories.

        Only the frames produced since the last cycle are projected and
        appended; warp resets are detected and clear the stale history. Returns
        the per-walker progress-coordinate ranges needed for binning.
        """
        from cowera.cv_history import CVHistory

        n = len(walkers)
        if self._cv_history is None or self._cv_history.n_walkers != n:
            self._cv_history = CVHistory(n, window=self.history_window)

        dranges = [None] * n
        for i in range(n):
            new_proj, drange, total, was_reset = self.distance.project_new_frames(
                i, folder, int(self._cv_history.offsets[i]))
            if was_reset:
                self._cv_history.reset(i)
            self._cv_history.extend(i, new_proj)
            self._cv_history.offsets[i] = total
            dranges[i] = drange
        return dranges

    def get_dist(self, walkers, folder, n_d, it, n_bins, max_bins):

        # Per-walker projection (image) onto the progress coordinate.
        images = [self.distance.get_proj_coord(walker.state)[0] for walker in walkers]

        # Full symmetric pairwise distance matrix, computed in a single
        # vectorized pass that reuses one cached topology/backbone selection
        # (previously this constructed two MDAnalysis Universes for *every*
        # walker pair, every cycle).
        dist_mat = self.distance.pairwise_distance_matrix(walkers)

        # Per-walker intensities + adaptive bin count.
        if self.use_cv_history:
            # Disk-free analysis: consume the in-memory accumulated histories.
            dranges = self._update_cv_histories(walkers, folder, n_d)
            projections = self._cv_history.as_list()
            dl, n_bins = self.distance.intensity_from_projections(
                projections, dranges, n_d=n_d, it=it,
                n_bins=n_bins, max_bins=max_bins,
                bin_increase_factor=self.bin_increase_factor,
                bin_decrease_factor=self.bin_decrease_factor)
        else:
            # Legacy path: re-read and re-project each walker's full DCD.
            dl, n_bins = self.distance.intensity_calculation(
                n_walkers=len(walkers), path=folder, n_d=n_d, it=it,
                n_bins=n_bins, max_bins=max_bins,
                bin_increase_factor=self.bin_increase_factor,
                bin_decrease_factor=self.bin_decrease_factor)

        return dl, [row for row in dist_mat], images, n_bins

    @log_call(include_args=[],
              include_result=False)

    def resample(self, walkers, cycle_id, n_bins, max_bins=125):
        """Resamples walkers based on REVO algorithm

        Parameters
        ----------
        walkers : list of walkers


        Returns
        -------
        resampled_walkers : list of resampled_walkers

        resampling_data : list of dict of str: value
            The resampling records resulting from the decisions.

        resampler_data :list of dict of str: value
            The resampler records resulting from the resampler actions.

        """

        #initialize the parameters
        num_walkers = len(walkers)
        walker_weights = [walker.weight for walker in walkers]
        num_walker_copies = [1 for i in range(num_walkers)]
        dcd_folder = self.dcd_folder
        cycle_id = cycle_id
        n_bins = n_bins
        max_bins = max_bins
        # calculate  distances
        distance_arr, distance_matrix, images, n_bins = self.get_dist(walkers,folder = dcd_folder , n_d = self.n_d, it = cycle_id,n_bins = n_bins,max_bins = max_bins)

        if self.increment == 1:
            # Closest walker distance
            cw_dist = np.max(images)
        else:
            cw_dist = np.min(images)

        # determine cloning and merging actions to be performed, by
        # maximizing the variation, i.e. the Decider
        resampling_data, variation, happen = self.decide(walker_weights, num_walker_copies, distance_arr, distance_matrix,images)


        # Per-cycle count of feature-read fallbacks (init-structure CV substituted
        # for a walker). Nonzero flags possible CV/intensity corruption this cycle.
        total_fallbacks = get_fallback_count()
        cv_fallbacks = total_fallbacks - self._prev_fallback_count
        self._prev_fallback_count = total_fallbacks

        file = open(f'{self.info_file_path}', 'a')
        file.write(f'Cycle: {cycle_id}'+'\t'+f'Clst walk. proj: {cw_dist}'+'\t'+f'Resampling happend: {happen}'+'\t'+f'n_bins: {n_bins}'+'\t'+f'CV_fallbacks: {cv_fallbacks}'+'\n')
        file.close()

        # convert the target idxs and decision_id to feature vector arrays
        for record in resampling_data:
            record['target_idxs'] = np.array(record['target_idxs'])
            record['decision_id'] = np.array([record['decision_id']])

        # update trajectory files according to the resampling data
        update_dcd_files(resampling_data, dcd_folder = dcd_folder)

        # mirror the clone/merge reorganization in the in-memory CV histories so
        # that next cycle's incremental update continues from the correct
        # parent history for each (possibly cloned) walker slot.
        if self.use_cv_history and self._cv_history is not None:
            self._cv_history.reindex(resampling_data)

        # actually do the cloning and merging of the walkers
        resampled_walkers = self.DECISION.action(walkers, [resampling_data])


        # flatten the distance matrix and give the number of walkers
        # as well for the resampler data, there is just one per cycle
        resampler_data = [{'distance_array' : distance_arr,
                           'num_walkers' : np.array([len(walkers)]),
                           'variation' : np.array([variation]),
                           'images' : np.ravel(np.array(images)),
                           'image_shape' : np.array(images[0].shape)}]

        # record the adaptive bin count for checkpointing/restart
        self._last_n_bins = n_bins

        return resampled_walkers, resampling_data, resampler_data, n_bins