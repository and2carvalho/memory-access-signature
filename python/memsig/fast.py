"""Bit-exact numba (njit) kernels of the pipeline.

Compiled counterparts of sorts.py, workloads.py and signal.py. The order and number of
`push` calls per kernel are IDENTICAL to the pure-Python port (verified bit for bit by
python/verify_kernels.py, and against output/atlas.json and output/crossdomain.json by
python/check_reproduction.py). They make the full grid (n up to 20,000, quadratic cases
included) run in about a minute.

Reservoir sampling is implemented inline, with the same logic as SignalStore.push.
"""

import time

import numpy as np
from numba import int64, njit
from numba.experimental import jitclass

from .inputs import generate_input

MASK32 = 0xFFFFFFFF

# ---------------------------------------------------------------------------
# Bit-exact mulberry32 (mirrors prng.py / src/prng.ts)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _to_int32(x):
    x &= MASK32
    if x & 0x80000000:
        return x - 0x100000000
    return x


@njit(cache=True)
def _ushr(a, n):
    return (a & MASK32) >> n


@njit(cache=True)
def _imul(a, b):
    return _to_int32(_to_int32(a) * _to_int32(b))


@njit(cache=True)
def _m32_next(state):
    a = _to_int32(state + 0x6D2B79F5)
    t = _to_int32(a ^ _ushr(a, 15))
    t = _imul(t, 1 | a)
    t = _to_int32((t + _imul(t ^ _ushr(t, 7), 61 | t)) ^ t)
    return ((_to_int32(t ^ _ushr(t, 14)) & MASK32) / 4294967296.0), a


# ---------------------------------------------------------------------------
# SignalStore in numba (order-preserving reservoir)
# ---------------------------------------------------------------------------

_fast_store_spec = [
    ("sig", int64[:]),
    ("pos", int64[:]),
    ("count", int64),
    ("stored", int64),
    ("max_stored", int64),
    ("state", int64),
]


@jitclass(_fast_store_spec)
class _FastStore:
    def __init__(self, max_stored, rng_seed):
        self.sig = np.zeros(max_stored, dtype=np.int64)
        self.pos = np.zeros(max_stored, dtype=np.int64)
        self.count = 0
        self.stored = 0
        self.max_stored = max_stored
        self.state = rng_seed & MASK32

    def push(self, v):
        t = self.count + 1
        self.count = t
        if self.stored < self.max_stored:
            self.sig[self.stored] = v
            self.pos[self.stored] = t
            self.stored += 1
            return
        r, self.state = _m32_next(self.state)
        r_int = int(r * t)
        if r_int < self.max_stored:
            self.sig[r_int] = v
            self.pos[r_int] = t


# ---------------------------------------------------------------------------
# Seven instrumented sorts (mirror sorts.py)
# ---------------------------------------------------------------------------


@njit(cache=True)
def insertion_sort_kernel(a, store):
    n = a.size
    for i in range(1, n):
        current = a[i]
        j = i - 1
        while j >= 0 and a[j] > current:
            store.push(j)       # read of A[j] in the comparison
            store.push(j + 1)   # write A[j+1] = A[j] (shift)
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = current


@njit(cache=True)
def merge_sort_kernel(a, store):
    n = a.size
    if n <= 1:
        return
    tmp = np.empty(n, dtype=np.int64)
    cap = 4 * n + 16
    st_lo = np.empty(cap, dtype=np.int64)
    st_hi = np.empty(cap, dtype=np.int64)
    st_st = np.empty(cap, dtype=np.int64)
    top = 0
    st_lo[top] = 0
    st_hi[top] = n
    st_st[top] = 0
    top += 1
    while top > 0:
        top -= 1
        lo = st_lo[top]
        hi = st_hi[top]
        state = st_st[top]
        if hi - lo <= 1:
            continue
        mid = (lo + hi) >> 1
        if state == 0:
            st_lo[top] = lo
            st_hi[top] = hi
            st_st[top] = 1
            top += 1
            st_lo[top] = mid
            st_hi[top] = hi
            st_st[top] = 0
            top += 1
            st_lo[top] = lo
            st_hi[top] = mid
            st_st[top] = 0
            top += 1
        else:
            i = lo
            j = mid
            k = lo
            while i < mid and j < hi:
                store.push(i)
                store.push(j)
                if a[i] <= a[j]:
                    tmp[k] = a[i]
                    i += 1
                else:
                    tmp[k] = a[j]
                    j += 1
                k += 1
            while i < mid:
                store.push(i)
                tmp[k] = a[i]
                i += 1
                k += 1
            while j < hi:
                store.push(j)
                tmp[k] = a[j]
                j += 1
                k += 1
            for p in range(lo, hi):
                store.push(p)
                a[p] = tmp[p]


@njit(cache=True)
def _quick_swap(a, store, x, y):
    if x == y:
        return
    store.push(x)
    store.push(y)
    tmp = a[x]
    a[x] = a[y]
    a[y] = tmp


@njit(cache=True)
def _quick_partition(a, store, lo, hi):
    mid = (lo + hi) >> 1
    if a[lo] > a[mid]:
        _quick_swap(a, store, lo, mid)
    if a[lo] > a[hi]:
        _quick_swap(a, store, lo, hi)
    if a[mid] > a[hi]:
        _quick_swap(a, store, mid, hi)
    p = mid
    _quick_swap(a, store, p, lo)
    pivot = a[lo]
    i = lo
    for j in range(lo + 1, hi + 1):
        store.push(j)
        if a[j] < pivot:
            i += 1
            _quick_swap(a, store, i, j)
    _quick_swap(a, store, lo, i)
    return i


@njit(cache=True)
def quick_sort_kernel(a, store):
    n = a.size
    cap = n + 16
    st_lo = np.empty(cap, dtype=np.int64)
    st_hi = np.empty(cap, dtype=np.int64)
    top = 0
    st_lo[top] = 0
    st_hi[top] = n - 1
    top += 1
    while top > 0:
        top -= 1
        lo = st_lo[top]
        hi = st_hi[top]
        if lo >= hi:
            continue
        p = _quick_partition(a, store, lo, hi)
        st_lo[top] = p + 1
        st_hi[top] = hi
        top += 1
        st_lo[top] = lo
        st_hi[top] = p - 1
        top += 1


@njit(cache=True)
def _heap_swap(a, store, x, y):
    if x == y:
        return
    store.push(x)
    store.push(y)
    tmp = a[x]
    a[x] = a[y]
    a[y] = tmp


@njit(cache=True)
def _heap_sift(a, store, start, size):
    i = start
    while True:
        l = 2 * i + 1
        r = 2 * i + 2
        largest = i
        if l < size:
            store.push(l)
            if a[l] > a[largest]:
                largest = l
        if r < size:
            store.push(r)
            if a[r] > a[largest]:
                largest = r
        if largest == i:
            break
        _heap_swap(a, store, i, largest)
        i = largest


@njit(cache=True)
def heap_sort_kernel(a, store):
    n = a.size
    for i in range((n >> 1) - 1, -1, -1):
        _heap_sift(a, store, i, n)
    for i in range(n - 1, 0, -1):
        _heap_swap(a, store, 0, i)
        _heap_sift(a, store, 0, i)


@njit(cache=True)
def _naive_swap(a, store, x, y):
    if x == y:
        return
    store.push(x)
    store.push(y)
    tmp = a[x]
    a[x] = a[y]
    a[y] = tmp


@njit(cache=True)
def quick_sort_naive_kernel(a, store):
    n = a.size
    cap = n + 16
    st_lo = np.empty(cap, dtype=np.int64)
    st_hi = np.empty(cap, dtype=np.int64)
    top = 0
    st_lo[top] = 0
    st_hi[top] = n - 1
    top += 1
    while top > 0:
        top -= 1
        lo = st_lo[top]
        hi = st_hi[top]
        if lo >= hi:
            continue
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            store.push(j)
            if a[j] < pivot:
                _naive_swap(a, store, i, j)
                i += 1
        _naive_swap(a, store, i, hi)
        p = i
        st_lo[top] = p + 1
        st_hi[top] = hi
        top += 1
        st_lo[top] = lo
        st_hi[top] = p - 1
        top += 1


@njit(cache=True)
def shell_sort_kernel(a, store):
    n = a.size
    h = 1
    while h < n / 3:
        h = 3 * h + 1
    while h >= 1:
        for i in range(h, n):
            current = a[i]
            j = i
            while j >= h and a[j - h] > current:
                store.push(j - h)
                store.push(j)
                a[j] = a[j - h]
                j -= h
            a[j] = current
        h = h // 3


@njit(cache=True)
def radix_sort_kernel(a, store):
    n = a.size
    if n <= 1:
        return
    max_val = 0
    for i in range(n):
        if a[i] > max_val:
            max_val = a[i]
    out = np.empty(n, dtype=np.int64)
    count = np.zeros(256, dtype=np.int64)
    exp = 1
    while exp <= max_val:
        for i in range(256):
            count[i] = 0
        for i in range(n):
            store.push(i)
            count[(a[i] // exp) % 256 & 255] += 1
        for i in range(1, 256):
            count[i] += count[i - 1]
        for i in range(n - 1, -1, -1):
            d = (a[i] // exp) % 256 & 255
            store.push(i)
            count[d] -= 1
            out[count[d]] = a[i]
        for i in range(n):
            store.push(i)
            a[i] = out[i]
        exp *= 256


SORT_KERNELS = {
    "insertion": insertion_sort_kernel,
    "merge": merge_sort_kernel,
    "quicksort": quick_sort_kernel,
    "quicksort_naive": quick_sort_naive_kernel,
    "heapsort": heap_sort_kernel,
    "shellsort": shell_sort_kernel,
    "radix": radix_sort_kernel,
}


@njit(cache=True)
def _is_sorted_np(a):
    for i in range(1, a.size):
        if a[i - 1] > a[i]:
            return False
    return True


def run_sort_fast(algo, a_np, max_stored=100_000, rng_seed=0):
    store = _FastStore(max_stored, rng_seed)
    t0 = time.perf_counter()
    SORT_KERNELS[algo](a_np, store)
    time_ms = (time.perf_counter() - t0) * 1000
    if not _is_sorted_np(a_np):
        raise RuntimeError(f"sort postcondition failed: {algo}")
    sampled = store.count > store.max_stored
    return {
        "events": store.count,
        "signal": store.sig[: store.stored].copy(),
        "positions": store.pos[: store.stored].copy() if sampled else None,
        "sampled": sampled,
        "timeMs": time_ms,
    }


# ---------------------------------------------------------------------------
# Four cross-domain workloads (mirror workloads.py)
# ---------------------------------------------------------------------------


@njit(cache=True)
def linear_reduce_kernel(a, store):
    acc = 0
    for i in range(a.size):
        store.push(i)
        acc += a[i]
    return acc


@njit(cache=True)
def tree_reduce_kernel(a, store):
    n = a.size
    for i in range(n - 1, 0, -1):
        parent = (i - 1) >> 1
        store.push(i)
        store.push(parent)
        a[parent] += a[i]
    return a[0]


@njit(cache=True)
def hash_probe_kernel(a, store, probe_seed):
    n = a.size
    state = probe_seed & MASK32
    acc = 0
    for _ in range(n):
        r, state = _m32_next(state)
        idx = int(r * n)
        store.push(idx)
        acc += a[idx]
    return acc


@njit(cache=True)
def self_join_kernel(a, store):
    n = a.size
    tau = 4
    matches = 0
    for i in range(n):
        ai = a[i]
        for j in range(n):
            store.push(i)
            store.push(j)
            d = ai - a[j]
            if d <= tau and d >= -tau:
                matches += 1
    return matches


def run_workload_fast(name, a_np, max_stored=100_000, rng_seed=0):
    store = _FastStore(max_stored, rng_seed)
    if name == "linear_reduce":
        result = linear_reduce_kernel(a_np, store)
        ok = result == int(a_np.sum())
    elif name == "tree_reduce":
        result = tree_reduce_kernel(a_np, store)
        ok = result > 0
    elif name == "hash_probe":
        result = hash_probe_kernel(a_np, store, 0x51ED270B ^ a_np.size)
        ok = result >= 0
    elif name == "self_join":
        result = self_join_kernel(a_np, store)
        ok = result >= a_np.size
    else:
        raise ValueError(f"carga desconhecida: {name}")
    if not ok:
        raise RuntimeError(f"workload postcondition failed: {name} (result={result})")
    sampled = store.count > store.max_stored
    return {
        "events": store.count,
        "signal": store.sig[: store.stored].copy(),
        "positions": store.pos[: store.stored].copy() if sampled else None,
        "sampled": sampled,
        "result": result,
        "timeMs": 0.0,
    }


# ---------------------------------------------------------------------------
# Fast builders (mirror build_atlas.py / build_crossdomain.py)
# ---------------------------------------------------------------------------


def _to_np(input_list):
    return np.array(input_list, dtype=np.int64)


def build_atlas(sizes=None, trials=5, base_seed=20260731, max_stored=100_000,
                quad_cap=20_000):
    from .inputs import PATTERNS
    from .signature import index_trajectory_drift, relative_density

    sizes = sizes or [1000, 2000, 5000, 10000, 20000]
    QUADRATIC = {"insertion", "quicksort_naive"}

    ref_events_by_n = {}
    for n in sizes:
        a = _to_np(generate_input(n, "aleatorio", base_seed))
        ref_events_by_n[n] = run_sort_fast("merge", a, max_stored,
                                           base_seed ^ 0x9E3779B9)["events"]

    rows = []
    for algo in SORT_KERNELS:
        for pattern in PATTERNS:
            for n in sizes:
                if algo in QUADRATIC and n > quad_cap:
                    continue
                per = []
                for t in range(trials):
                    seed = base_seed + t
                    a = _to_np(generate_input(n, pattern, seed))
                    r = run_sort_fast(algo, a, max_stored, seed ^ 0x9E3779B9)
                    drift = index_trajectory_drift(r["signal"], r["positions"], n)
                    per.append({"events": r["events"], "drift": drift,
                                "sampled": r["sampled"]})
                events = round(sum(p["events"] for p in per) / trials)
                drift = sum(p["drift"] for p in per) / trials
                D = relative_density(events, ref_events_by_n[n])
                rows.append({
                    "algo": algo, "pattern": pattern, "n": n,
                    "events": events,
                    "D": round(D, 4),
                    "drift": round(drift, 3),
                    "refEvents": ref_events_by_n[n],
                    "sampled": per[0]["sampled"],
                    "trials": [{"events": p["events"], "drift": round(p["drift"], 3)}
                               for p in per],
                })

    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "method": "signature (D = relative access density, δ = index-trajectory drift); no resonant filter",
        "densityReference": "merge/aleatorio at the same n (E_ref)",
        "params": {"sizes": sizes, "trials": trials, "baseSeed": base_seed,
                   "maxStored": max_stored, "quadCap": quad_cap},
        "rows": rows,
    }


def build_crossdomain(sizes=None, base_seed=20260731, max_stored=100_000,
                      quad_cap=20_000):
    from .signature import index_trajectory_drift, relative_density

    D_INFLATED = 5
    DRIFT_ZERO_BAND = 10
    sizes = sizes or [1000, 2000, 5000, 10000, 20000]
    QUADRATIC_WORKLOADS = {"self_join"}

    ref_by_n = {}
    for n in sizes:
        a = _to_np(generate_input(n, "aleatorio", base_seed))
        ref_by_n[n] = run_sort_fast("merge", a, max_stored,
                                    base_seed ^ 0x9E3779B9)["events"]

    families = {
        "linear_reduce": ("forward-scan", "H1"),
        "tree_reduce": ("root-return", "H2"),
        "self_join": ("quadratic-overload", "H3"),
        "hash_probe": ("random-access", "H4"),
    }

    rows = []
    for name in ["linear_reduce", "tree_reduce", "self_join", "hash_probe"]:
        family, hypothesis = families[name]
        for n in sizes:
            if name in QUADRATIC_WORKLOADS and n > quad_cap:
                continue
            a = _to_np(generate_input(n, "aleatorio", base_seed))
            r = run_workload_fast(name, a, max_stored, base_seed ^ 0x9E3779B9)
            events = r["events"]
            drift = index_trajectory_drift(r["signal"], r["positions"], n)
            D = relative_density(events, ref_by_n[n])
            rows.append({
                "workload": name, "family": family, "hypothesis": hypothesis,
                "n": n, "events": events, "D": round(D, 4), "drift": round(drift, 3),
                "refEvents": ref_by_n[n], "sampled": r["sampled"],
                "regime": "inflada" if D > D_INFLATED else "normal",
                "driftSign": "≈0" if abs(drift) < DRIFT_ZERO_BAND
                             else ("+" if drift > 0 else "-"),
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
