"""Bit-exact check: numba kernels (memsig/fast.py) versus the pure-Python port."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from memsig import generate_input, ALGORITHMS, WORKLOADS
from memsig.prng import mulberry32
from memsig.signal import SignalStore
from memsig.signature import index_trajectory_drift
from memsig.fast import run_sort_fast, run_workload_fast

MAX_STORED = 100_000


def pure_run_sort(algo, pattern, n, seed):
    input_ = generate_input(n, pattern, seed)
    store = SignalStore(MAX_STORED, mulberry32(seed ^ 0x9E3779B9), 0, n * n)
    ALGORITHMS[algo](input_[:], store.push)
    s = store.signal
    return {
        "events": s["originalLength"],
        "signal": s["signal"],
        "positions": s["positions"],
        "sampled": s["sampled"],
    }


def pure_run_workload(name, n, seed):
    input_ = generate_input(n, "aleatorio", seed)
    store = SignalStore(MAX_STORED, mulberry32(seed ^ 0x9E3779B9), 0, 2 * n * n)
    wl = WORKLOADS[name]
    result = wl["run"](input_[:], store.push)
    if not wl["verify"](input_, result):
        raise RuntimeError(f"workload postcondition failed: {name}")
    s = store.signal
    return {
        "events": s["originalLength"],
        "signal": s["signal"],
        "positions": s["positions"],
        "sampled": s["sampled"],
    }


def check_sort(algo, pattern, n, seed):
    pure = pure_run_sort(algo, pattern, n, seed)
    a_np = np.array(generate_input(n, pattern, seed), dtype=np.int64)
    fast = run_sort_fast(algo, a_np, MAX_STORED, seed ^ 0x9E3779B9)
    errs = []
    if pure["events"] != fast["events"]:
        errs.append(f"events {pure['events']} vs {fast['events']}")
    if pure["sampled"] != fast["sampled"]:
        errs.append(f"sampled {pure['sampled']} vs {fast['sampled']}")
    if not np.array_equal(np.asarray(pure["signal"]), np.asarray(fast["signal"])):
        errs.append("signal differs")
    pp = pure["positions"]
    fp = fast["positions"]
    if (pp is None) != (fp is None):
        errs.append(f"positions None-ness {pp is None} vs {fp is None}")
    elif pp is not None and not np.array_equal(np.asarray(pp), np.asarray(fp)):
        errs.append("positions differ")
    if errs:
        print(f"  MISMATCH {algo}/{pattern}/n={n}/seed={seed}: {'; '.join(errs)}")
        return 1
    return 0


def check_workload(name, n, seed):
    pure = pure_run_workload(name, n, seed)
    a_np = np.array(generate_input(n, "aleatorio", seed), dtype=np.int64)
    fast = run_workload_fast(name, a_np, MAX_STORED, seed ^ 0x9E3779B9)
    errs = []
    if pure["events"] != fast["events"]:
        errs.append(f"events {pure['events']} vs {fast['events']}")
    if not np.array_equal(np.asarray(pure["signal"]), np.asarray(fast["signal"])):
        errs.append("signal differs")
    pp = pure["positions"]
    fp = fast["positions"]
    if (pp is None) != (fp is None):
        errs.append("positions None-ness")
    elif pp is not None and not np.array_equal(np.asarray(pp), np.asarray(fp)):
        errs.append("positions differ")
    if errs:
        print(f"  MISMATCH {name}/n={n}: {'; '.join(errs)}")
        return 1
    return 0


def main():
    t0 = time.time()
    bad = 0
    print("sorts (seed=20260731):")
    for algo in ALGORITHMS:
        for pattern in ["aleatorio", "quase_ordenado", "inverso"]:
            bad += check_sort(algo, pattern, 1000, 20260731)
            bad += check_sort(algo, pattern, 2000, 20260731 + 1)
    print(f"workloads (seed=20260731):")
    for name in WORKLOADS:
        bad += check_workload(name, 1000, 20260731)
        bad += check_workload(name, 2000, 20260731)
    # stress case: naive QuickSort on reversed input (O(n²) degeneration)
    print("O(n²) degeneration — quicksort_naive/inverso, n=3000:")
    bad += check_sort("quicksort_naive", "inverso", 3000, 20260731)
    bad += check_sort("insertion", "quase_ordenado", 5000, 20260731)
    print(f"\n{'OK — bit-exact' if bad == 0 else f'FAILED: {bad} mismatches'} ({time.time()-t0:.1f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
