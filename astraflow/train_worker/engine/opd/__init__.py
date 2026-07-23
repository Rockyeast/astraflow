"""On-policy distillation training primitives."""

from .functional import (
    align_and_compute_opd_advantages,
    compute_reverse_kl_advantages,
)

__all__ = [
    "align_and_compute_opd_advantages",
    "compute_reverse_kl_advantages",
]
