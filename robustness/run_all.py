"""Robustness analysis of the (D, δ) signature.

Each test addresses a methodological objection that a critical reader could raise:

  T1 baseline_collapse  — δ versus trivial order-aware trend estimators (half/third/
                          end-point differences, Theil-Sen, raw OLS) and versus an
                          order-blind statistic (μ, the mean accessed index). Reports
                          rank correlation, agreement of decisions (sign, ≈0 band,
                          H1–H4 verdicts) and the Heapsort vs Merge/QuickSort separation.
  T2 order_ablation     — permutes the temporal order of events: does δ collapse to ~0?
  T3 robustness         — stability of the sign of δ with respect to the number of
                          bands b and to the reservoir size `max_stored`.
  T4 boundary_tent      — boundary of the signature: a synthetic "tent" trajectory
                          (zero net trend, strong temporal structure) and the δ≈0
                          overlap between Radix (multi-pass scan) and hash_probe (random
                          access), resolved by the lag-1 autocorrelation ρ₁.

The harness does not modify the main pipeline: it imports `python/memsig` (bit-exact
kernels) and regenerates traces with the atlas seeds (BASE_SEED = 20260731).

Outputs: robustness/output/t{1..4}_*.json and robustness/output/summary.json
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from memsig.fast import SORT_KERNELS, run_sort_fast, run_workload_fast  # noqa: E402
from memsig.inputs import PATTERNS, generate_input  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
BASE_SEED = 20260731
MAX_STORED = 100_000
QUADRATIC = {"insertion", "quicksort_naive"}
SIZES = [2000, 5000, 10000, 20000]
WORKLOADS = ["linear_reduce", "tree_reduce", "self_join", "hash_probe"]
TRIALS = 5


# ---------------------------------------------------------------------------
# Signature tools (same definitions as the main pipeline, re-implemented here)
# ---------------------------------------------------------------------------


def band_means_from(signal, positions, n, bands):
    pairs = sorted(
        (positions[i] if positions is not None else i, v) for i, v in enumerate(signal)
    )
    L = len(pairs)
    means = []
    for b in range(bands):
        lo = math.floor((b / bands) * L)
        hi = math.floor(((b + 1) / bands) * L)
        s = 0.0
        for i in range(lo, hi):
            s += pairs[i][1]
        means.append(s / max(1, hi - lo))
    return np.asarray(means, dtype=float)


def ordered_values(signal, positions):
    if positions is None:
        return np.asarray(signal, dtype=float)
    pairs = sorted(zip(positions, signal))
    return np.asarray([v for _, v in pairs], dtype=float)


def _slope(xs, ys):
    xm = xs.mean()
    ym = ys.mean()
    denom = ((xs - xm) ** 2).sum()
    return ((xs - xm) * (ys - ym)).sum() / (denom if denom else 1e-12)


def ols(means, n, bands):
    m = means.size
    if m < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, m)
    return 1000.0 * _slope(x, means / max(1, n - 1)) / bands


def drift_from(signal, positions, n, bands):
    if len(signal) < 8:
        return 0.0
    return ols(band_means_from(signal, positions, n, bands), n, bands)


def drift_raw(stream, n, bands):
    L = stream.size
    if L < 8:
        return 0.0
    x = np.linspace(0.0, 1.0, L)
    return 1000.0 * _slope(x, stream / max(1, n - 1)) / bands


def alt_endpoints(means, n, bands):
    m = means.size
    if m < 2:
        return 0.0
    return 1000.0 * (means[-1] - means[0]) / max(1, n - 1) / bands


def alt_halves(means, n, bands):
    m = means.size
    if m < 2:
        return 0.0
    mid = m // 2
    return 1000.0 * (means[mid:].mean() - means[:mid].mean()) / max(1, n - 1) / bands


def alt_thirds(means, n, bands):
    m = means.size
    if m < 3:
        return 0.0
    parts = np.array_split(means, 3)
    return 1000.0 * (parts[2].mean() - parts[0].mean()) / max(1, n - 1) / bands


def alt_theil(means, n, bands):
    m = means.size
    if m < 2:
        return 0.0
    y = means / max(1, n - 1)
    slopes = []
    for i in range(m):
        for j in range(i + 1, m):
            dx = (j - i) / (m - 1)
            slopes.append((y[j] - y[i]) / dx)
    return 1000.0 * float(np.median(slopes)) / bands


# ---------------------------------------------------------------------------
# Corpus (same seeds as the atlas; traces are regenerated, output/ is not touched)
# ---------------------------------------------------------------------------


def build_corpus():
    rows = []  # one entry per cell (algo/pattern/n) with aggregated statistics

    for algo in SORT_KERNELS:
        for pattern in PATTERNS:
            for n in SIZES:
                if algo in QUADRATIC and n > 10000:
                    continue  # quadratic cases limited to n ≤ 10⁴ for wall-clock time
                acc = {k: [] for k in
                       ("d_ols", "d_ep", "d_half", "d_third", "d_theil", "d_raw", "mu")}
                events_list = []
                for t in range(TRIALS):
                    seed = BASE_SEED + t
                    a = np.asarray(generate_input(n, pattern, seed), dtype=np.int64)
                    r = run_sort_fast(algo, a, MAX_STORED, seed ^ 0x9E3779B9)
                    sig, pos, ev = r["signal"], r["positions"], r["events"]
                    means = band_means_from(sig, pos, n, 10)
                    stream = ordered_values(sig, pos)
                    acc["d_ols"].append(ols(means, n, 10))
                    acc["d_ep"].append(alt_endpoints(means, n, 10))
                    acc["d_half"].append(alt_halves(means, n, 10))
                    acc["d_third"].append(alt_thirds(means, n, 10))
                    acc["d_theil"].append(alt_theil(means, n, 10))
                    acc["d_raw"].append(drift_raw(stream, n, 10))
                    acc["mu"].append(stream.mean() / max(1, n - 1))
                    events_list.append(ev)
                rows.append({
                    "kind": "sort", "algo": algo, "pattern": pattern, "n": n,
                    "events": int(round(np.mean(events_list))),
                    **{k: float(np.mean(v)) for k, v in acc.items()},
                })

    for name in WORKLOADS:
        for n in SIZES:
            if name == "self_join" and n > 10000:
                continue
            a = np.asarray(generate_input(n, "aleatorio", BASE_SEED), dtype=np.int64)
            r = run_workload_fast(name, a, MAX_STORED, BASE_SEED ^ 0x9E3779B9)
            sig, pos = r["signal"], r["positions"]
            means = band_means_from(sig, pos, n, 10)
            stream = ordered_values(sig, pos)
            rows.append({
                "kind": "workload", "algo": name, "pattern": "aleatorio", "n": n,
                "events": int(r["events"]),
                "d_ols": ols(means, n, 10),
                "d_ep": alt_endpoints(means, n, 10),
                "d_half": alt_halves(means, n, 10),
                "d_third": alt_thirds(means, n, 10),
                "d_theil": alt_theil(means, n, 10),
                "d_raw": drift_raw(stream, n, 10),
                "mu": float(stream.mean() / max(1, n - 1)),
            })

    # D with the same reference as the atlas (merge/aleatorio per n)
    ref_by_n = {}
    for n in SIZES:
        a = np.asarray(generate_input(n, "aleatorio", BASE_SEED), dtype=np.int64)
        ref_by_n[n] = run_sort_fast("merge", a, MAX_STORED,
                                    BASE_SEED ^ 0x9E3779B9)["events"]
    for r in rows:
        r["D"] = r["events"] / ref_by_n[r["n"]]
    return rows, ref_by_n


# ---------------------------------------------------------------------------
# T1 — baseline collapse
# ---------------------------------------------------------------------------


def t1_baseline_collapse(rows):
    alts = ["d_ep", "d_half", "d_third", "d_theil", "d_raw", "mu"]
    stats = {}

    def spearman(a, b):
        from scipy.stats import spearmanr
        rho, p = spearmanr(a, b)
        return float(rho), float(p)

    base = np.asarray([r["d_ols"] for r in rows], dtype=float)
    for alt in alts:
        vals = np.asarray([r[alt] for r in rows], dtype=float)
        rho, p = spearman(base, vals)

        strong = [r for r in rows if abs(r["d_ols"]) >= 10]
        weak = [r for r in rows if abs(r["d_ols"]) < 10]

        def sign_agree(sub):
            if not sub:
                return float("nan")
            n_ok = sum(1 for r in sub if np.sign(r["d_ols"]) == np.sign(r[alt]))
            return n_ok / len(sub)

        def zeroband_agree(sub):
            if not sub:
                return float("nan")
            n_ok = sum(1 for r in sub
                       if (abs(r["d_ols"]) < 10) == (abs(r[alt]) < 10))
            return n_ok / len(sub)

        stats[alt] = {
            "spearman_vs_ols": rho,
            "sign_agreement_all": sign_agree(rows),
            "sign_agreement_strong": sign_agree(strong),
            "sign_agreement_weak": sign_agree(weak),
            "zero_band_agreement_all": zeroband_agree(rows),
            "material_disagreements": [
                {
                    "algo": r["algo"], "pattern": r["pattern"], "n": r["n"],
                    "d_ols": r["d_ols"], alt: r[alt],
                }
                for r in rows
                if (np.sign(r["d_ols"]) != np.sign(r[alt]))
                or ((abs(r["d_ols"]) < 10) != (abs(r[alt]) < 10))
            ],
        }

    # --- Cross-domain verdicts with the SAME pre-registered thresholds, per n ---
    D_INFLATED, ZERO = 5.0, 10.0
    HYP_OF = {"linear_reduce": "H1", "tree_reduce": "H2",
              "self_join": "H3", "hash_probe": "H4"}

    def hyp_pass(feature_name, hyp, r):
        f = r[feature_name]
        if hyp == "H1":
            return f > 40 and r["D"] < 1
        if hyp == "H2":
            return f < 0
        if hyp == "H3":
            return r["D"] > D_INFLATED
        if hyp == "H4":
            return abs(f) < ZERO
        raise KeyError(hyp)

    verdict_table = []
    for r in rows:
        if r["kind"] != "workload":
            continue
        hyp = HYP_OF[r["algo"]]
        verdict_table.append({
            "workload": r["algo"], "n": r["n"],
            "d_ols": r["d_ols"], "base_pass": bool(hyp_pass("d_ols", hyp, r)),
            "alts_pass": {alt: bool(hyp_pass(alt, hyp, r)) for alt in alts},
        })

    verdicts = {"n": len(verdict_table), "rows": verdict_table}
    for alt in alts:
        n_agree = sum(1 for v in verdict_table if v["base_pass"] == v["alts_pass"][alt])
        verdicts[f"agreement_{alt}"] = n_agree / len(verdict_table)

    # --- Heapsort vs Merge/QuickSort separation ---
    # uses per-trial executions (re-runs the cells at n = 10k)
    n10 = 10000
    feats = {"d_ols": [], "mu": [], "D": []}
    fam = {"heapsort": "heap", "merge": "merge", "quicksort": "quick"}
    label = []
    ref10 = {}
    a = np.asarray(generate_input(n10, "aleatorio", BASE_SEED), dtype=np.int64)
    ref10 = run_sort_fast("merge", a, MAX_STORED, BASE_SEED ^ 0x9E3779B9)["events"]
    for algo in ["heapsort", "merge", "quicksort"]:
        for t in range(TRIALS):
            seed = BASE_SEED + t
            a = np.asarray(generate_input(n10, "aleatorio", seed), dtype=np.int64)
            r = run_sort_fast(algo, a, MAX_STORED, seed ^ 0x9E3779B9)
            means = band_means_from(r["signal"], r["positions"], n10, 10)
            stream = ordered_values(r["signal"], r["positions"])
            feats["d_ols"].append(ols(means, n10, 10))
            feats["mu"].append(stream.mean() / max(1, n10 - 1))
            feats["D"].append(r["events"] / ref10)
            label.append(fam[algo])

    separation = {}
    for f, vals in feats.items():
        vals = np.asarray(vals)
        lab = np.asarray(label)
        heap = vals[lab == "heap"]
        rest = vals[lab != "heap"]
        pooled = np.std(vals, ddof=1)
        separation[f] = {
            "mean_heapsort": float(heap.mean()),
            "mean_merge_quick": float(rest.mean()),
            "abs_diff": float(abs(heap.mean() - rest.mean())),
            "pooled_std": float(pooled),
            "effect_size_diff_std": float(abs(heap.mean() - rest.mean()) / pooled),
        }

    stats["separation_heapsort_vs_mergequick"] = separation
    stats["verdicts"] = verdicts
    return stats


# ---------------------------------------------------------------------------
# T2 — order ablation (permutation)
# ---------------------------------------------------------------------------


def t2_order_ablation():
    runs = [
        ("heapsort", "aleatorio"),
        ("merge", "aleatorio"),
        ("quicksort", "aleatorio"),
        ("insertion", "inverso"),
        ("quicksort_naive", "quase_ordenado"),
        ("radix", "aleatorio"),
    ]
    n = 10000
    results = {}
    rng = np.random.default_rng(20260000)
    for algo, pattern in runs:
        seed = BASE_SEED
        a = np.asarray(generate_input(n, pattern, seed), dtype=np.int64)
        r = run_sort_fast(algo, a, MAX_STORED, seed ^ 0x9E3779B9)
        stream = ordered_values(r["signal"], r["positions"])
        d_true = drift_from(r["signal"], r["positions"], n, 10)
        perms = []
        signs_same = 0
        for s in range(60):
            perm = rng.permutation(stream)
            d_p = drift_from(perm, None, n, 10)
            perms.append(d_p)
            if np.sign(d_p) == np.sign(d_true):
                signs_same += 1
        perms = np.asarray(perms)
        results[f"{algo}/{pattern}"] = {
            "d_true": d_true,
            "n_events_stream": int(stream.size),
            "mean_abs_perm": float(np.abs(perms).mean()),
            "p95_abs_perm": float(np.percentile(np.abs(perms), 95)),
            "sign_match_fraction": signs_same / len(perms),
            "d_true_over_mean_abs_perm": float(d_true / max(1e-9, np.abs(perms).mean())),
        }
    return results


# ---------------------------------------------------------------------------
# T3 — robustness (number of bands b; reservoir size max_stored)
# ---------------------------------------------------------------------------


def t3_robustness(rows):
    # (a) sign / ≈0-band per number of bands b ∈ {4,8,10,16,32,64}. The corpus keeps
    # only aggregates, so a small subset of traces is regenerated and re-banded here.
    bands_list = [4, 8, 10, 16, 32, 64]
    n = 10000
    cells = []
    for algo in SORT_KERNELS:
        for pattern in PATTERNS:
            seed = BASE_SEED
            a = np.asarray(generate_input(n, pattern, seed), dtype=np.int64)
            r = run_sort_fast(algo, a, MAX_STORED, seed ^ 0x9E3779B9)
            cells.append((algo, pattern, r["signal"], r["positions"]))

    band_results = {"bands": bands_list, "per_cell": []}
    flips_sign = flips_zero = 0
    for algo, pattern, sig, pos in cells:
        per_b = {}
        for b in bands_list:
            means = band_means_from(sig, pos, n, b)
            per_b[b] = ols(means, n, b)
        signs = {b: np.sign(per_b[b]) for b in bands_list}
        zeros = {b: abs(per_b[b]) < 10 for b in bands_list}
        sign_flip = len(set(signs.values())) > 1
        zero_flip = len(set(zeros.values())) > 1
        flips_sign += sign_flip
        flips_zero += zero_flip
        band_results["per_cell"].append({
            "algo": algo, "pattern": pattern,
            "drift_by_bands": {str(b): round(per_b[b], 3) for b in bands_list},
            "sign_flips_across_bands": sign_flip,
            "zero_band_flips_across_bands": zero_flip,
        })
    band_results["n_cells"] = len(cells)
    band_results["n_sign_flips_across_bands"] = flips_sign
    band_results["n_zero_band_flips_across_bands"] = flips_zero

    # (b) reservoir: self_join n=20000 (heavily sampled) and heapsort n=20000
    reservoir_results = {}
    for name, kind in [("self_join", "workload"), ("heapsort", "sort")]:
        n2 = 20000
        a = np.asarray(generate_input(n2, "aleatorio", BASE_SEED), dtype=np.int64)
        series = []
        for ms in [1000, 5000, 20000, 100000]:
            if kind == "sort":
                r = run_sort_fast(name, a, ms, BASE_SEED ^ 0x9E3779B9)
            else:
                r = run_workload_fast(name, a, ms, BASE_SEED ^ 0x9E3779B9)
            d = drift_from(r["signal"], r["positions"], n2, 10)
            series.append({"max_stored": ms, "events": int(r["events"]),
                           "stored": int(r["signal"].size), "drift": round(d, 3)})
        reservoir_results[f"{name}(n={n2})"] = series
    return {"band_stability": band_results, "reservoir": reservoir_results}


# ---------------------------------------------------------------------------
# T4 — boundary: tent trajectory (δ≈0, strong structure) and the δ≈0 overlap, resolved by ρ₁
# ---------------------------------------------------------------------------


def rho1(means):
    if means.size < 4:
        return 0.0
    a = means[:-1]
    b = means[1:]
    denom = a.std() * b.std()
    if denom == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def tent_stream(n, passes):
    """'Tent' trajectory spanning the WHOLE trace: `passes` up/down cycles (0→h→0).
    The net trend is zero by construction (OLS ≈ 0), while the band means trace a
    triangle, giving a high lag-1 autocorrelation — structure that δ cannot represent."""
    h = n // 2
    L = 2 * n
    t = np.linspace(0.0, 2.0 * passes, L, endpoint=False)
    tri = 1.0 - np.abs((t % 2.0) - 1.0)   # triangular wave 0..1..0
    return (tri * h).astype(float)


def t4_boundary(rows):
    n = 10000
    out = {}

    # (a) δ≈0 overlap: radix (multi-pass) vs hash_probe (random) — separated by ρ₁
    conflate = []
    for name, kind, fam in [("radix", "sort", "multipass"), ("hash_probe", "workload", "random")]:
        a = np.asarray(generate_input(n, "aleatorio", BASE_SEED), dtype=np.int64)
        if kind == "sort":
            r = run_sort_fast(name, a, MAX_STORED, BASE_SEED ^ 0x9E3779B9)
        else:
            r = run_workload_fast(name, a, MAX_STORED, BASE_SEED ^ 0x9E3779B9)
        sig, pos = r["signal"], r["positions"]
        means64 = band_means_from(sig, pos, n, 64)
        d10 = drift_from(sig, pos, n, 10)
        conflate.append({
            "workload": name, "family": fam,
            "delta_10bands": round(d10, 3),
            "rho1_64bands": round(rho1(means64), 3),
        })
    out["conflation_delta0"] = conflate

    # (b) tent: δ≈0 (the linear fit is blind to it) but high ρ₁ — the unresolved domain, measured
    tent_rows = []
    for passes in [1, 2, 4]:
        stream = tent_stream(n, passes)
        d = drift_from(stream, None, n, 10)
        m64 = band_means_from(stream, None, n, 64)
        m10 = band_means_from(stream, None, n, 10)
        tent_rows.append({
            "passes": passes,
            "events": int(stream.size),
            "delta_10bands": round(d, 3),
            "rho1_64bands": round(rho1(m64), 3),
            "rho1_10bands": round(rho1(m10), 3),
        })
    out["tent_zero_drift"] = tent_rows
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    t0 = time.time()
    rows, ref_by_n = build_corpus()
    t1 = time.time()
    print(f"[corpus] {len(rows)} cells in {t1 - t0:.1f}s")

    r1 = t1_baseline_collapse(rows)
    t2 = time.time()
    print(f"[T1] baseline_collapse ok in {t2 - t1:.1f}s")
    r2 = t2_order_ablation()
    t3 = time.time()
    print(f"[T2] order_ablation ok in {t3 - t2:.1f}s")
    r3 = t3_robustness(rows)
    t4 = time.time()
    print(f"[T3] robustness ok in {t4 - t3:.1f}s")
    r4 = t4_boundary(rows)
    print(f"[T4] boundary ok in {time.time() - t4:.1f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [("t1_baseline_collapse", r1), ("t2_order_ablation", r2),
                       ("t3_robustness", r3), ("t4_boundary", r4)]:
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))

    summary = {
        "corpus_cells": len(rows),
        "ref_events_by_n": ref_by_n,
        "t1_spearman_mu_vs_ols": r1.get("mu", {}).get("spearman_vs_ols"),
        "t1_spearman_half_vs_ols": r1.get("d_half", {}).get("spearman_vs_ols"),
        "t1_separation": r1.get("separation_heapsort_vs_mergequick"),
        "t2_ablation_collapses": {
            k: v["d_true_over_mean_abs_perm"] for k, v in r2.items()
        },
        "t3_band_sign_flips": r3["band_stability"]["n_sign_flips_across_bands"],
        "t3_band_zero_flips": r3["band_stability"]["n_zero_band_flips_across_bands"],
        "t3_reservoir": r3["reservoir"],
        "t4_tent": r4["tent_zero_drift"],
        "t4_conflation": r4["conflation_delta0"],
        "total_seconds": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[summary] {time.time() - t0:.1f}s total. Outputs written to {OUT}")


if __name__ == "__main__":
    main()
