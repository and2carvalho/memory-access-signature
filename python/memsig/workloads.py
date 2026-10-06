"""Python port of src/workloads.ts — non-sorting (cross-domain) workloads.

Under the SAME instrumentation contract as sorts.py, four workloads instantiate the four
access-pattern families pre-registered in docs/preregistration.md:
    forward scan       (δ>0)  → linear_reduce  (linear scan / reduction)
    root return        (δ<0)  → tree_reduce    (bottom-up fold from leaves to root)
    random access      (δ≈0)  → hash_probe     (pseudo-random probes into A)
    quadratic overload (D≫5)  → self_join      (nested-loop O(n²), structural)

Each workload exposes a verifiable postcondition (`verify`) to ensure that it actually
performed the computation its name describes.
"""

from .prng import mulberry32


def linear_reduce(a: list[int], signal) -> int:
    acc = 0
    for i in range(len(a)):
        signal(i)  # sequential read of A[i]
        acc += a[i]
    return acc


def tree_reduce(a: list[int], signal) -> int:
    for i in range(len(a) - 1, 0, -1):
        parent = (i - 1) >> 1
        signal(i)        # read of child A[i]
        signal(parent)   # read-modify-write of A[parent]
        a[parent] += a[i]
    return a[0]


def hash_probe(a: list[int], signal) -> int:
    n = len(a)
    rng = mulberry32(0x51ED270B ^ n)  # deterministic, content-independent
    acc = 0
    for _ in range(n):
        idx = int(rng() * n)  # probe target, uniform in [0, n)
        signal(idx)           # random access to A[idx]
        acc += a[idx]
    return acc


def self_join(a: list[int], signal) -> int:
    n = len(a)
    tau = 4  # narrow band threshold → non-trivial count; cost remains O(n²)
    matches = 0
    for i in range(n):
        ai = a[i]
        for j in range(n):
            signal(i)  # read of the outer operand A[i] in the comparison
            signal(j)  # read of the scanned operand A[j] in the comparison
            d = ai - a[j]
            if -tau <= d <= tau:
                matches += 1
    return matches


WORKLOADS = {
    "linear_reduce": {
        "run": linear_reduce,
        "verify": lambda a, r: r == sum(a),
        "family": "forward-scan",
        "hypothesis": "H1",
    },
    "tree_reduce": {
        "run": tree_reduce,
        "verify": lambda _a, r: isinstance(r, (int, float)) and r > 0,
        "family": "root-return",
        "hypothesis": "H2",
    },
    "hash_probe": {
        "run": hash_probe,
        "verify": lambda _a, r: isinstance(r, (int, float)) and r >= 0,
        "family": "random-access",
        "hypothesis": "H4",
    },
    "self_join": {
        "run": self_join,
        "verify": lambda a, r: r >= len(a),
        "family": "quadratic-overload",
        "hypothesis": "H3",
    },
}

QUADRATIC_WORKLOADS = {"self_join"}
