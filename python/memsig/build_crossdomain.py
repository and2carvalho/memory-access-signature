"""Python port of src/build_crossdomain.ts — cross-domain validation.

Measures four non-sorting workloads (workloads.py) under the SAME thresholds and the SAME
density reference as the sorting atlas (a transfer test, without recalibration).
Hypotheses and predictions are stated in docs/preregistration.md.
"""

import json
import os
import time

from .prng import mulberry32
from .inputs import generate_input
from .signal import SignalStore
from .sorts import ALGORITHMS
from .signature import index_trajectory_drift, relative_density
from .workloads import WORKLOADS, QUADRATIC_WORKLOADS

D_INFLATED = 5  # inflated-load boundary, calibrated on the sorting atlas
DRIFT_ZERO_BAND = 10


def event_bound(name: str, n: int) -> int:
    if name == "self_join":
        return 2 * n * n + 4 * n
    if name == "tree_reduce":
        return 2 * n + 4
    return n + 4


def ref_events(n: int, seed: int, max_stored: int) -> int:
    input_ = generate_input(n, "aleatorio", seed)
    store = SignalStore(max_stored, mulberry32(seed ^ 0x9E3779B9), 0,
                        n * math_ceil_log2(n) * 6)
    ALGORITHMS["merge"](input_[:], store.push)
    return store.signal["originalLength"]


def math_ceil_log2(n: int) -> int:
    return (n - 1).bit_length()


def classify_regime(D: float) -> str:
    return "inflada" if D > D_INFLATED else "normal"


def classify_drift(delta: float) -> str:
    if abs(delta) < DRIFT_ZERO_BAND:
        return "≈0"
    return "+" if delta > 0 else "-"


def build_crossdomain(sizes=None, base_seed=20260731, max_stored=100_000,
                      quad_cap=20_000):
    sizes = sizes or [1000, 2000, 5000, 10000, 20000]

    ref_by_n = {n: ref_events(n, base_seed, max_stored) for n in sizes}

    rows = []
    for name, wl in WORKLOADS.items():
        for n in sizes:
            if name in QUADRATIC_WORKLOADS and n > quad_cap:
                continue
            input_ = generate_input(n, "aleatorio", base_seed)
            store = SignalStore(max_stored, mulberry32(base_seed ^ 0x9E3779B9), 0,
                                event_bound(name, n))
            result = wl["run"](input_[:], store.push)
            if not wl["verify"](input_, result):
                raise RuntimeError(f"workload postcondition failed: {name}/n={n} (result={result})")
            s = store.signal
            events = s["originalLength"]
            drift = index_trajectory_drift(s["signal"], s["positions"], n)
            D = relative_density(events, ref_by_n[n])
            rows.append({
                "workload": name, "family": wl["family"], "hypothesis": wl["hypothesis"],
                "n": n, "events": events, "D": round(D, 4), "drift": round(drift, 3),
                "refEvents": ref_by_n[n], "sampled": s["sampled"],
                "regime": classify_regime(D), "driftSign": classify_drift(drift),
            })

    last = {}
    for r in rows:
        last[r["workload"]] = r
    verdicts = {
        "H1": {"pass": last["linear_reduce"]["drift"] > 40 and last["linear_reduce"]["D"] < 1,
               "drift": last["linear_reduce"]["drift"], "D": last["linear_reduce"]["D"]},
        "H2": {"pass": last["tree_reduce"]["drift"] < 0,
               "drift": last["tree_reduce"]["drift"], "D": last["tree_reduce"]["D"]},
        "H3": {"pass": last["self_join"]["D"] > D_INFLATED,
               "drift": last["self_join"]["drift"], "D": last["self_join"]["D"]},
        "H4": {"pass": abs(last["hash_probe"]["drift"]) < DRIFT_ZERO_BAND,
               "drift": last["hash_probe"]["drift"], "D": last["hash_probe"]["D"]},
    }

    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "method": "cross-domain transfer of (D, δ) — non-sort workloads under sort-calibrated thresholds",
        "densityReference": "merge/aleatorio at the same n (E_ref), identical to atlas",
        "thresholds": {"D_inflated": D_INFLATED, "drift_zero_band": DRIFT_ZERO_BAND},
        "params": {"sizes": sizes, "baseSeed": base_seed, "maxStored": max_stored,
                   "quadCap": quad_cap},
        "preregistration": "docs/preregistration.md",
        "verdicts": verdicts,
        "rows": rows,
    }


def main():
    payload = build_crossdomain()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(root, "output", "crossdomain.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("── Pre-registered hypotheses ──")
    for h, v in payload["verdicts"].items():
        print(f"  {h}: {'SUPPORTED' if v['pass'] else 'NOT SUPPORTED'}  (D={v['D']}, δ={v['drift']}‰)")
    print(f"OK — {len(payload['rows'])} rows written to {out}")


if __name__ == "__main__":
    main()
