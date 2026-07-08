import os
import os.path as osp
import pickle
import logging

from wepy.reporter.reporter import Reporter

# File name templates for the two halves of a checkpoint: the walker states
# (needed to re-seed the ensemble) and a small metadata sidecar (cycle index and
# adaptive bin count) needed to resume resampling coherently. The per-walker CV
# histories are intentionally NOT serialized: they rebuild deterministically from
# the persisted ``walker_*.dcd`` trajectories on the first post-restart cycle.
WALKERS_TEMPLATE = "walkers_cycle_{}.pkl"
CHECKPOINT_TEMPLATE = "checkpoint_cycle_{}.pkl"


def _cycle_of(fname, template):
    """Return the integer cycle index encoded in ``fname`` for ``template``, or None."""
    prefix, suffix = template.split("{}")
    if fname.startswith(prefix) and fname.endswith(suffix):
        middle = fname[len(prefix):len(fname) - len(suffix)] if suffix else fname[len(prefix):]
        if middle.isdigit():
            return int(middle)
    return None


def latest_checkpoint(save_dir):
    """Locate the newest *complete* checkpoint in ``save_dir``.

    A complete checkpoint has both a ``walkers_cycle_<k>.pkl`` and its matching
    ``checkpoint_cycle_<k>.pkl`` metadata sidecar (guarding against a job killed
    mid-write). Pure filesystem logic (no MD deps) so it is unit-testable.

    Returns
    -------
    (walkers_path, checkpoint_path, cycle_idx) or None
    """
    if not osp.isdir(save_dir):
        return None
    names = os.listdir(save_dir)
    walker_cycles = {c for c in (_cycle_of(n, WALKERS_TEMPLATE) for n in names) if c is not None}
    ckpt_cycles = {c for c in (_cycle_of(n, CHECKPOINT_TEMPLATE) for n in names) if c is not None}
    complete = walker_cycles & ckpt_cycles
    if not complete:
        return None
    k = max(complete)
    return (osp.join(save_dir, WALKERS_TEMPLATE.format(k)),
            osp.join(save_dir, CHECKPOINT_TEMPLATE.format(k)),
            k)


class WalkersPickleReporter(Reporter):
    """Periodically pickles walker states so a killed run can be resumed.

    Each backup writes two files: the walker states and a small metadata sidecar
    (``cycle_idx`` and the current adaptive ``n_bins``). Set ``wipe_on_init`` to
    False on restart so previously written checkpoints are preserved.
    """

    def __init__(self, save_dir='./', freq=100, num_backups=2, wipe_on_init=True):
        # the directory to save the pickles in
        self.save_dir = save_dir
        # the frequency of cycles to backup the walkers as a pickle
        self.backup_freq = freq
        # the number of sets of walker pickles to keep, this will keep
        # the last `num_backups`
        self.num_backups = num_backups
        # whether init() clears existing pickles (False when resuming a run)
        self.wipe_on_init = wipe_on_init
        # set at init() from the simulation components; used to snapshot n_bins
        self.resampler = None

    def init(self, resampler=None, **kwargs):
        # keep a handle to the resampler so we can snapshot its adaptive bin count
        self.resampler = resampler
        # make sure the save_dir exists
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # On a fresh run, clear stale pickles. On restart (wipe_on_init=False)
        # keep them so the checkpoint that seeded this run is not destroyed.
        elif self.wipe_on_init:
            for pkl_fname in os.listdir(self.save_dir):
                os.remove(osp.join(self.save_dir, pkl_fname))

    def _remove_if_exists(self, fname):
        path = osp.join(self.save_dir, fname)
        if osp.exists(path):
            os.remove(path)

    def report(self, cycle_idx=None, new_walkers=None,
               **kwargs):
        # total number of cycles completed
        n_cycles = cycle_idx + 1
        # if the cycle is on the frequency backup walkers to a pickle
        if n_cycles % self.backup_freq == 0:
            # walker states
            with open(osp.join(self.save_dir, WALKERS_TEMPLATE.format(cycle_idx)), 'wb') as wf:
                pickle.dump(new_walkers, wf)

            # metadata sidecar (written second so latest_checkpoint only sees a
            # complete pair). Snapshot the resampler's current adaptive bin count.
            n_bins = getattr(self.resampler, '_last_n_bins', None)
            meta = {'cycle_idx': cycle_idx, 'n_bins': n_bins}
            with open(osp.join(self.save_dir, CHECKPOINT_TEMPLATE.format(cycle_idx)), 'wb') as mf:
                pickle.dump(meta, mf)

            # remove old checkpoints beyond the retained window
            if (cycle_idx // self.backup_freq) >= self.num_backups:
                old_idx = cycle_idx - self.num_backups * self.backup_freq
                self._remove_if_exists(WALKERS_TEMPLATE.format(old_idx))
                self._remove_if_exists(CHECKPOINT_TEMPLATE.format(old_idx))
