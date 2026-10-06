#!/usr/bin/env python3
"""
Scaling exponent of D with n.

For each (algorithm, pattern) series, fits the power law

    D(n) = C · n^ν   ⇔   log10 D = log10 C + ν · log10 n

by least squares over the atlas sizes (5 points: 1k..20k), reporting ν, R² and the
95% confidence interval of the exponent (Student's t, 3 degrees of freedom).

Reads  output/atlas.json  and writes  output/scaling.json.

Interpretation. D is defined as a RATIO to the O(n log n) reference at the same n.
Hence, by construction of the normalization:
    - O(n log n)  ⇒ D ~ const                  ⇒ ν ≈ 0
    - O(n²)       ⇒ D ~ n²/(n log n) = n/log n  ⇒ ν ≈ 0.87 (not 1.0: the log n in the
                    denominator lowers the effective slope over the measured range)
This script checks the latter by fitting ν to the reference function n/log2(n) itself
(field "expected_quadratic_nu"). At the extremes, ν therefore only RECOVERS the
complexity classes already encoded by D; it is confirmatory and does not indicate a
phase transition or a universal law. Its incremental value lies in what the binary
threshold D > 5 does not resolve:
    (i)   INTERMEDIATE regimes (e.g. shellsort, 0 < ν < 0.87);
    (ii)  INPUT SENSITIVITY of adaptive algorithms (e.g. insertion: ν ≈ 0 on nearly
          sorted input versus ≈ 0.88 on random input — same code);
    (iii) pattern-invariant scaling (e.g. radix: identical ν across patterns).
"""
import json
import os
import sys
from datetime import datetime, timezone
from math import sqrt, log2, log10

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = os.path.join(ROOT, "output", "atlas.json")
OUT = os.path.join(ROOT, "output", "scaling.json")

# Two-tailed 95% critical t values, indexed by degrees of freedom (n_points - 2).
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def loglog_fit(n, y_val):
    """Fits log10(y) = a + ν log10(n). Returns (ν, R², 95% CI half-width of the slope)."""
    x = np.log10(np.asarray(n, float))
    y = np.log10(np.asarray(y_val, float))
    A = np.vstack([np.ones_like(x), x]).T
    (a, nu), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([a, nu])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = len(x) - 2
    if dof > 0:
        s2 = ss_res / dof
        sxx = float(np.sum((x - x.mean()) ** 2))
        se = sqrt(s2 / sxx) if sxx > 0 else float("nan")
        ci = T_CRIT.get(dof, 2.0) * se
    else:
        ci = float("nan")
    return float(nu), float(r2), float(ci)


def classify(nu, ci):
    """Descriptive regime label derived from the measured exponent (not from the algorithm)."""
    lo, hi = nu - ci, nu + ci
    if hi < 0.15:
        return "constant (O(n log n))"
    if lo > 0.6:
        return "inflated (~quadratic)"
    return "intermediate"


ALGO_COLOR = {
    "insertion": "#d62728", "merge": "#1f77b4", "quicksort": "#2ca02c",
    "quicksort_naive": "#9467bd", "heapsort": "#ff7f0e", "shellsort": "#8c564b",
    "radix": "#17becf",
}


def make_figure(rows, out_rows, exp_quad_nu, pattern="aleatorio"):
    """D versus n (log-log) with the fitted power law and ν ± CI annotated.
    Skipped when matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figure (numbers are in scaling.json)")
        return
    figdir = os.path.join(ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    fits = {(r["algo"], r["pattern"]): r for r in out_rows}
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for algo, color in ALGO_COLOR.items():
        sub = sorted([r for r in rows if r["algo"] == algo and r["pattern"] == pattern],
                     key=lambda r: r["n"])
        if len(sub) < 3:
            continue
        n = np.array([r["n"] for r in sub], float)
        D = np.array([r["D"] for r in sub], float)
        f = fits.get((algo, pattern))
        ax.plot(n, D, "o", color=color, ms=6, zorder=3)
        # fitted line
        x = np.log10(n)
        coef = np.polyfit(x, np.log10(D), 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(10 ** xs, 10 ** (coef[1] + coef[0] * xs), "-", color=color, lw=1.6,
                label=f"{algo}: ν={f['nu']:+.2f}±{f['ci95']:.2f} (R²={f['r2']:.2f})")
    ax.axhline(5.0, color="crimson", lw=1, ls=":", zorder=1)
    ax.text(1100, 5.6, "inflated-load threshold D=5", color="crimson", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("D — relative access density (log scale)")
    ax.set_title(
        "Scaling exponent of D (random input)\n"
        f"O(n²) → ν≈{exp_quad_nu:.2f} (= n/log n, not 1); O(n log n) → ν≈0; "
        "shellsort intermediate; radix ~O(n)")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
              title="fitted power law")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(figdir, "fig2b_scaling_exponent.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {p}")


def main():
    if not os.path.exists(ATLAS):
        sys.exit(f"atlas not found: {ATLAS} (run: node src/build_atlas.ts)")
    rows = json.load(open(ATLAS))["rows"]

    ns = sorted(set(r["n"] for r in rows))
    # Expected exponent of the reference under the normalization of D:
    #   an O(n²) algorithm has D(n) ∝ n²/(n log2 n) = n/log2 n.
    ref = [n / log2(n) for n in ns]
    exp_quad_nu, exp_quad_r2, _ = loglog_fit(ns, ref)
    # A pure O(n) workload (e.g. radix, linear scan) has D ∝ n/(n log2 n) = 1/log2 n.
    ref_lin = [1.0 / log2(n) for n in ns]
    exp_lin_nu, _, _ = loglog_fit(ns, ref_lin)

    algos = sorted(set(r["algo"] for r in rows))
    pats = sorted(set(r["pattern"] for r in rows))
    out_rows = []
    print(f"{'algo':17s}{'pattern':16s}{'nu':>8s}{'±95%':>8s}{'R2':>8s}   regime")
    print("-" * 74)
    for algo in algos:
        for pat in pats:
            sub = sorted([r for r in rows if r["algo"] == algo and r["pattern"] == pat],
                         key=lambda r: r["n"])
            if len(sub) < 3:
                continue
            n = [r["n"] for r in sub]
            D = [r["D"] for r in sub]
            nu, r2, ci = loglog_fit(n, D)
            reg = classify(nu, ci)
            out_rows.append({
                "algo": algo, "pattern": pat, "nu": round(nu, 4),
                "ci95": round(ci, 4), "r2": round(r2, 4),
                "n_points": len(sub), "regime": reg,
            })
            print(f"{algo:17s}{pat:16s}{nu:8.3f}{ci:8.3f}{r2:8.4f}   {reg}")

    make_figure(rows, out_rows, exp_quad_nu)

    result = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "output/atlas.json",
        "model": "log10 D = log10 C + nu * log10 n",
        "n_values": ns,
        "reference_note": (
            "D is normalized by an O(n log n) reference; the expected exponent of an "
            "O(n^2) workload under this ratio is nu(n/log2 n), not 1.0."
        ),
        "expected_quadratic_nu": round(exp_quad_nu, 4),
        "expected_quadratic_r2": round(exp_quad_r2, 4),
        "expected_linear_nu": round(exp_lin_nu, 4),
        "rows": out_rows,
    }
    json.dump(result, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("-" * 74)
    print(f"O(n^2) reference under the normalization of D -> expected nu = {exp_quad_nu:.3f} (R2={exp_quad_r2:.4f})")
    print(f"O(n)   reference under the normalization of D -> expected nu = {exp_lin_nu:.3f}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
