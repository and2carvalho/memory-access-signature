#!/usr/bin/env python3
"""
Analysis of the (D, δ) signature over the sorting atlas.

Reads  output/atlas.json  (produced by src/build_atlas.ts) and produces:
  - Cluster separability in the (log10 D, δ) plane via the silhouette score
    (k-means, k = 2..6).
  - Cross-n stability of the discriminators: consistency of the sign of δ and of the
    load regime of D for each (algorithm, pattern) series across all n.
  - Figures: (D, δ) plane; D versus n (log-log); δ versus n; silhouette versus k.
  - output/analysis.json with the metrics.

Usage:  python analysis/signature_analysis.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = os.path.join(ROOT, "output", "atlas.json")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Descriptive decision boundary, observed on this corpus (not a universal constant).
D_INFLATED = 5.0   # inflated-load boundary (degeneration / quadratic workload)

PATTERN_MARKER = {"aleatorio": "o", "quase_ordenado": "^", "inverso": "s"}
ALGO_COLOR = {
    "insertion": "#d62728", "merge": "#1f77b4", "quicksort": "#2ca02c",
    "quicksort_naive": "#9467bd", "heapsort": "#ff7f0e", "shellsort": "#8c564b",
    "radix": "#17becf",
}


def load_rows():
    with open(ATLAS) as f:
        data = json.load(f)
    return data["rows"], data["params"]


def behavioral_family(D, drift):
    """Behavioural label obtained by an explicit rule over the signature."""
    if D >= D_INFLATED:
        return "inflated load (D≥5)"
    if drift < 0:
        return "root-return (δ<0)"
    return "forward-scan (δ>0)"


def silhouette_analysis(rows):
    """Silhouette of (unsupervised) k-means in the standardized (log10 D, δ) plane."""
    X = np.array([[np.log10(max(r["D"], 1e-6)), r["drift"]] for r in rows], dtype=float)
    Xs = (X - X.mean(0)) / X.std(0)
    out = {}
    best = (None, -1)
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        s = float(silhouette_score(Xs, km.labels_))
        out[k] = s
        if s > best[1]:
            best = (k, s)
    return out, best, Xs


def cross_n_stability(rows):
    """Consistency of the sign of δ and of the load regime per (algorithm, pattern) across n."""
    keys = sorted({(r["algo"], r["pattern"]) for r in rows})
    report = []
    for algo, pat in keys:
        sub = sorted([r for r in rows if r["algo"] == algo and r["pattern"] == pat],
                     key=lambda r: r["n"])
        drifts = [r["drift"] for r in sub]
        Ds = [r["D"] for r in sub]
        signs = {np.sign(round(d, 1)) for d in drifts if abs(d) > 1e-6}
        sign_consistent = len(signs) <= 1
        regime = {"inflated" if D >= D_INFLATED else "normal" for D in Ds}
        regime_consistent = len(regime) == 1
        report.append({
            "algo": algo, "pattern": pat,
            "n_values": [r["n"] for r in sub],
            "drift_range": [round(min(drifts), 1), round(max(drifts), 1)],
            "drift_sign_consistent": sign_consistent,
            "D_range": [round(min(Ds), 4), round(max(Ds), 4)],
            "D_regime_consistent": regime_consistent,
            "regime": sorted(regime),
        })
    return report


def fig_plane(rows, n_focus=10000):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    seen = set()
    for r in rows:
        if r["n"] != n_focus:
            continue
        c = ALGO_COLOR[r["algo"]]
        m = PATTERN_MARKER[r["pattern"]]
        lbl = r["algo"] if r["algo"] not in seen else None
        seen.add(r["algo"])
        ax.scatter(r["D"], r["drift"], c=c, marker=m, s=90,
                   edgecolors="k", linewidths=0.5, label=lbl, zorder=3)
    ax.axhline(0, color="grey", lw=1, ls="--", zorder=1)
    ax.axvline(D_INFLATED, color="crimson", lw=1, ls=":", zorder=1)
    ax.text(D_INFLATED * 1.1, ax.get_ylim()[1] * 0.92, "inflated load →",
            color="crimson", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("D — relative access density (log scale)")
    ax.set_ylabel("δ — index-trajectory drift (‰)")
    ax.set_title(f"Signature plane (D, δ) — n={n_focus}\n"
                 "colour = algorithm · marker = input (○ random △ nearly sorted □ reversed)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="algorithm")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig1_signature_plane.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_scaling_D(rows):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for algo in ALGO_COLOR:
        sub = sorted([r for r in rows if r["algo"] == algo and r["pattern"] == "aleatorio"],
                     key=lambda r: r["n"])
        if not sub:
            continue
        ax.plot([r["n"] for r in sub], [r["D"] for r in sub], "-o",
                color=ALGO_COLOR[algo], label=algo, ms=5)
    ax.axhline(D_INFLATED, color="crimson", lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("D — relative access density (log scale)")
    ax.set_title("Scaling of D with n (random input)\n"
                 "inflated load grows ~linearly; O(n log n) remains ~constant")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig2_scaling_D.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_drift_n(rows):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for algo in ALGO_COLOR:
        sub = sorted([r for r in rows if r["algo"] == algo and r["pattern"] == "aleatorio"],
                     key=lambda r: r["n"])
        if not sub:
            continue
        ax.plot([r["n"] for r in sub], [r["drift"] for r in sub], "-o",
                color=ALGO_COLOR[algo], label=algo, ms=5)
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("δ — index-trajectory drift (‰)")
    ax.set_title("Stability of the sign of δ with n (random input)\n"
                 "Heapsort remains negative; the others remain non-negative")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig3_drift_vs_n.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_silhouette(sil):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ks = sorted(sil)
    ax.plot(ks, [sil[k] for k in ks], "-o", color="#333")
    kbest = max(sil, key=sil.get)
    ax.scatter([kbest], [sil[kbest]], c="crimson", s=90, zorder=3,
               label=f"best k={kbest} (S={sil[kbest]:.3f})")
    ax.set_xlabel("k (number of clusters, k-means)")
    ax.set_ylabel("silhouette score")
    ax.set_title("Cluster structure in the (log D, δ) plane")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig4_silhouette.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    if not os.path.exists(ATLAS):
        sys.exit(f"atlas not found: {ATLAS} (run: node src/build_atlas.ts)")
    rows, params = load_rows()

    sil, best, _ = silhouette_analysis(rows)
    stab = cross_n_stability(rows)

    # Silhouette of the behavioural partition (labels from the explicit rule).
    X = np.array([[np.log10(max(r["D"], 1e-6)), r["drift"]] for r in rows], dtype=float)
    Xs = (X - X.mean(0)) / X.std(0)
    fam = [behavioral_family(r["D"], r["drift"]) for r in rows]
    fam_ids = {f: i for i, f in enumerate(sorted(set(fam)))}
    fam_labels = np.array([fam_ids[f] for f in fam])
    fam_sil = float(silhouette_score(Xs, fam_labels)) if len(fam_ids) > 1 else float("nan")

    figs = [fig_plane(rows), fig_scaling_D(rows), fig_drift_n(rows), fig_silhouette(sil)]

    n_sign_ok = sum(1 for s in stab if s["drift_sign_consistent"])
    n_reg_ok = sum(1 for s in stab if s["D_regime_consistent"])

    result = {
        "params": params,
        "silhouette_kmeans": sil,
        "best_k": {"k": best[0], "silhouette": round(best[1], 4)},
        "behavioral_family_silhouette": round(fam_sil, 4),
        "behavioral_families": sorted(fam_ids),
        "cross_n_stability": {
            "series_total": len(stab),
            "drift_sign_consistent": n_sign_ok,
            "D_regime_consistent": n_reg_ok,
            "detail": stab,
        },
        "figures": [os.path.relpath(p, ROOT) for p in figs],
    }
    out = os.path.join(ROOT, "output", "analysis.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=== Silhouette (k-means, log D × δ plane) ===")
    for k in sorted(sil):
        mark = "  <- best" if k == best[0] else ""
        print(f"  k={k}: S={sil[k]:.3f}{mark}")
    print(f"\nSilhouette of the behavioural partition (3 families): S={fam_sil:.3f}")
    print(f"  families: {sorted(fam_ids)}")
    print(f"\n=== Cross-n stability ({len(stab)} algorithm × pattern series) ===")
    print(f"  sign of δ consistent across n: {n_sign_ok}/{len(stab)}")
    print(f"  load regime (inflated/normal) consistent: {n_reg_ok}/{len(stab)}")
    print(f"\nFigures: {[os.path.basename(p) for p in figs]}")
    print(f"Metrics: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
