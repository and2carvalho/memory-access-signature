"""Python port of src/build_atlas.ts — atlas driver for the (D, δ) signature.

For each (algorithm × pattern × n × trial) the driver generates the input from a fixed
seed, runs the instrumented algorithm with order-preserving reservoir sampling, and
computes (D, δ) in memory. Density reference: E(ref) = merge/aleatorio at the SAME n.
Quadratic algorithms (insertion, quicksort_naive) are skipped above `quad_cap`.
"""

import json
import math
import os
import time

from .prng import mulberry32
from .inputs import generate_input, PATTERNS
from .signal import SignalStore
from .sorts import ALGORITHMS
from .signature import index_trajectory_drift, relative_density

HISTOGRAM_BUCKETS = 0
QUADRATIC = {"insertion", "quicksort_naive"}


def expected_length_bound(algo: str, n: int) -> int:
    log2n = math.ceil(math.log2(max(2, n)))
    if algo == "insertion":
        bound = n * n
    elif algo == "quicksort_naive":
        bound = n * n
    elif algo == "merge":
        bound = n * log2n * 4
    elif algo == "quicksort":
        bound = n * log2n * 3
    elif algo == "heapsort":
        bound = n * log2n * 4
    elif algo == "shellsort":
        bound = n * log2n * log2n * 4
    elif algo == "radix":
        bound = n * 16
    else:
        bound = n
    return math.ceil(bound * 1.5)


def is_sorted(a: list[int]) -> bool:
    return all(a[i - 1] <= a[i] for i in range(1, len(a)))


def run_once(algo: str, pattern: str, n: int, seed: int, max_stored: int):
    input_ = generate_input(n, pattern, seed)
    store = SignalStore(max_stored, mulberry32(seed ^ 0x9E3779B9),
                        HISTOGRAM_BUCKETS, expected_length_bound(algo, n))
    a = input_[:]
    t0 = time.perf_counter()
    ALGORITHMS[algo](a, store.push)
    time_ms = (time.perf_counter() - t0) * 1000
    if not is_sorted(a):
        raise RuntimeError(f"sort postcondition failed: {algo}/{pattern}/n={n}/seed={seed}")
    s = store.signal
    drift = index_trajectory_drift(s["signal"], s["positions"], n)
    return {
        "events": s["originalLength"],
        "drift": drift,
        "timeMs": time_ms,
        "sampled": s["sampled"],
    }


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def build_atlas(sizes=None, trials=5, base_seed=20260731, max_stored=100_000,
                quad_cap=20_000):
    sizes = sizes or [1000, 2000, 5000, 10000, 20000]

    # Global density reference per n: events of merge/aleatorio.
    ref_events_by_n = {}
    for n in sizes:
        ref_events_by_n[n] = run_once("merge", "aleatorio", n, base_seed, max_stored)["events"]

    rows = []
    for algo in ALGORITHMS:
        for pattern in PATTERNS:
            for n in sizes:
                if algo in QUADRATIC and n > quad_cap:
                    continue
                per = [run_once(algo, pattern, n, base_seed + t, max_stored)
                       for t in range(trials)]
                events = round(mean(r["events"] for r in per))
                drift = mean(r["drift"] for r in per)
                D = relative_density(events, ref_events_by_n[n])
                rows.append({
                    "algo": algo, "pattern": pattern, "n": n,
                    "events": events,
                    "D": round(D, 4),
                    "drift": round(drift, 3),
                    "refEvents": ref_events_by_n[n],
                    "sampled": per[0]["sampled"],
                    "trials": [{"events": r["events"], "drift": round(r["drift"], 3)} for r in per],
                })

    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "method": "signature (D = relative access density, δ = index-trajectory drift); no resonant filter",
        "densityReference": "merge/aleatorio at the same n (E_ref)",
        "params": {"sizes": sizes, "trials": trials, "baseSeed": base_seed,
                   "maxStored": max_stored, "quadCap": quad_cap},
        "rows": rows,
    }


def main():
    import sys
    sizes = [1000, 2000, 5000, 10000, 20000]
    payload = build_atlas(sizes=sizes)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(root, "output", "atlas.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"OK — {len(payload['rows'])} rows written to {out}")


if __name__ == "__main__":
    main()
