"""memsig — Python reference implementation of the (D, δ) memory-access signature.

Mirrors the TypeScript pipeline in src/*.ts and reproduces its outputs bit for bit.
Usage:

    from memsig import mulberry32, build_atlas, build_crossdomain
"""

from .prng import mulberry32, int_between
from .inputs import generate_input, PATTERNS
from .signal import SignalStore
from .sorts import ALGORITHMS, insertion_sort, merge_sort, quick_sort, heap_sort, quick_sort_naive, shell_sort, radix_sort
from .workloads import WORKLOADS, linear_reduce, tree_reduce, hash_probe, self_join
from .signature import index_trajectory_drift, relative_density
from .build_atlas import build_atlas, run_once
from .build_crossdomain import build_crossdomain

__all__ = [
    "mulberry32", "int_between",
    "generate_input", "PATTERNS",
    "SignalStore",
    "ALGORITHMS", "insertion_sort", "merge_sort", "quick_sort", "heap_sort",
    "quick_sort_naive", "shell_sort", "radix_sort",
    "WORKLOADS", "linear_reduce", "tree_reduce", "hash_probe", "self_join",
    "index_trajectory_drift", "relative_density",
    "build_atlas", "run_once", "build_crossdomain",
]
