"""CoWERA-Committor: a learned committor as the live WE resampling criterion.

This is an **opt-in** extension subpackage. It must not be imported by the core
``cowera`` package so that the published CoWERA path keeps running without any
machine-learning dependencies. Heavy/optional backends (e.g. torch / GVP-GNN)
are confined here and imported lazily.

Near-term milestone (M1) provides a dependency-free, fully unit-tested core:

* :mod:`cowera_committor.featurizer`     -- SE(3)-invariant features
* :mod:`cowera_committor.committor_model`-- pure-numpy MLP committor (interface
                                            stable for a future GVP-GNN backend)
* :mod:`cowera_committor.committor_data` -- WE-weighted training buffers
* :mod:`cowera_committor.toy_systems`    -- analytic validation potentials
"""

from cowera_committor.featurizer import (
    IdentityFeaturizer,
    DistanceFeaturizer,
)
from cowera_committor.committor_model import CommittorModel, MLPCommittor
from cowera_committor.committor_data import (
    BoundaryBuffer,
    SemigroupBuffer,
    ShootingBuffer,
)
from cowera_committor.committor_metric import (
    CommittorDistances,
    augment_merge_distance,
    committor_displacement,
    suppress_tse_merges,
)
from cowera_committor.phase_manager import (
    BootstrapPhaseManager,
    STRUCTURAL, SEMIGROUP, SHOOTING, MATURE,
)
from cowera_committor.bootstrap import structural_interpolants
from cowera_committor.committor_train import (
    extract_segment_training_data,
    assemble_training_data,
    retrain_committor,
)

# NOTE: ``CommittorResampler`` is intentionally not imported here because it
# depends on the ``wepy`` stack. Import it directly where that stack is present:
#     from cowera_committor.committor_resampler import CommittorResampler

__all__ = [
    "IdentityFeaturizer",
    "DistanceFeaturizer",
    "CommittorModel",
    "MLPCommittor",
    "BoundaryBuffer",
    "SemigroupBuffer",
    "ShootingBuffer",
    "CommittorDistances",
    "augment_merge_distance",
    "committor_displacement",
    "suppress_tse_merges",
    "BootstrapPhaseManager",
    "STRUCTURAL", "SEMIGROUP", "SHOOTING", "MATURE",
    "structural_interpolants",
    "extract_segment_training_data",
    "assemble_training_data",
    "retrain_committor",
]
