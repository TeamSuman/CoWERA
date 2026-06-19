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

__all__ = [
    "IdentityFeaturizer",
    "DistanceFeaturizer",
    "CommittorModel",
    "MLPCommittor",
    "BoundaryBuffer",
    "SemigroupBuffer",
    "ShootingBuffer",
]
