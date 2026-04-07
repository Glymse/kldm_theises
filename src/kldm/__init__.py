"""
KLDM package.

Keep package-level imports minimal so subpackages like ``kldm.data`` can be
imported without pulling in unfinished training or sampling surfaces.
"""

__version__ = "0.1.0"
from .sample_evaluation.sample_evaluation import (
    SampleEvaluationResult,
    decode_atom_types,
    decode_lattice,
    evaluate_sample,
)

__all__ = [
    "SampleEvaluationResult",
    "decode_atom_types",
    "decode_lattice",
    "evaluate_sample",
]
