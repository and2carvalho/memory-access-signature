# Robustness analysis

This directory tests the `(D, δ)` signature against the strongest methodological
objections that can be raised about the role of δ. Each objection is translated into a
quantitative test in [`run_all.py`](run_all.py). The harness does **not** modify the main
pipeline: it imports the bit-exact kernels of [`python/memsig`](../python/memsig) and
regenerates traces with the atlas seeds (`BASE_SEED = 20260731`).

```bash
python robustness/run_all.py   # ~90 s; writes robustness/output/*.json
```

## Objections under test

1. **Is the drift estimator essential?** If trivial trend estimators reproduce every
   decision, the 10-band regression is incidental; and if an *order-blind* statistic
   separates the algorithms as well as δ does, the claim that temporal order is required
   does not hold. → **T1**
2. **Does δ measure order at all?** Destructive control: permuting the temporal order of
   the events should collapse δ to ≈ 0. → **T2**
3. **Is δ sensitive to analysis parameters?** Number of bands `b` and reservoir size. → **T3**
4. **What can δ not see?** As a linear fit, δ is blind by construction to trajectories
   with zero net trend. How large is that blind region? → **T4**

Corpus: 93 cells — 7 sorting algorithms × 3 input patterns plus 4 cross-domain
workloads, n ∈ {2k, 5k, 10k, 20k} (quadratic cases capped at n ≤ 10k), 5 trials per cell.

## Results

### T1 — δ versus alternative estimators

| Estimator | Spearman ρ vs δ | Sign agreement (strong cases) | Agreement on the \|δ\| < 10 band | Agreement on H1–H4 verdicts |
|---|---:|---:|---:|---:|
| `d_ep` end-point difference | 0.928 | 1.000 | 0.860 | 15/15 |
| `d_half` difference of halves | 0.983 | 1.000 | 0.849 | 15/15 |
| `d_third` difference of thirds | 0.990 | 1.000 | 0.914 | 15/15 |
| `d_theil` Theil–Sen | 0.914 | 1.000 | 0.925 | 15/15 |
| `d_raw` OLS without banding | **0.997** | 1.000 | **1.000** | 15/15 |
| **`μ` order-blind (mean index)** | **0.482** | **0.743** | **0.204** | **7/15** |

Heapsort vs Merge/QuickSort separation at n = 10⁴, in pooled standard deviations:
δ = 2.05σ, **μ = 2.05σ**, D = 1.74σ.

**Interpretation.**

- **The estimator is interchangeable.** Every trivial order-aware estimator — including
  OLS without banding (`d_raw`, ρ = 0.997, 100% agreement on the ≈ 0 band) — reproduces
  all cross-domain verdicts under the same pre-registered thresholds. The 10-band
  regression adds no information: the quantity of interest is the **direction of the
  low-frequency trend**, not the particular estimator.
- **The Heapsort vs Merge/QuickSort separation does not require temporal order.** The
  mean accessed index μ — a moment of the marginal *index* histogram, insensitive to
  order — separates the two groups with the same effect size as δ (2.05σ). What
  distinguishes them at this point is *where* the accesses concentrate (Heapsort favours
  low indices), not their order. The genuinely temporal contribution of δ appears in the
  **forward vs backward trend** cases (H1/H2), where μ fails (0/8 agreement) and every
  order-aware estimator succeeds (8/8).

### T2 — Order ablation by permutation

| Cell (n = 10⁴) | δ (true order) | mean \|δ\| under permutation | ratio | sign agreement |
|---|---:|---:|---:|---:|
| heapsort / random | −20.5 | 0.141 | **145×** | 0.52 |
| merge / random | +72.9 | 0.193 | **378×** | 0.50 |
| quicksort / random | +71.3 | 0.226 | **315×** | 0.53 |
| insertion / reversed | +35.7 | 0.182 | **196×** | 0.53 |
| quicksort_naive / nearly sorted | −35.7 | 0.199 | **180×** | 0.48 |
| radix / random | +3.1 | 0.254 | 12× | 0.47 |

Under permutation of the event order (same multiset of indices), δ collapses to
0.14–0.25 and its sign becomes random (≈ 0.5), while the true δ lies 12× to 378× above
the permutation baseline. **δ is an order statistic**, demonstrated destructively.

### T3 — Sensitivity to analysis parameters

- **Number of bands** `b ∈ {4, 8, 10, 16, 32, 64}` (21 cells, n = 10⁴): the **sign** of δ
  is stable in 19/21 cells; the two flips are Radix (nearly sorted / reversed) at `b = 4`,
  cells already in the δ ≈ 0 region. The **magnitude**, however, scales approximately as
  1/b (Insertion / random: 97.7 at `b = 4` → 8.1 at `b = 64`), and membership in the
  |δ| < 10 band changes with `b` in 10/21 cells (e.g. Heapsort / random: −40 at `b = 4`,
  −3.6 at `b = 64`).
- **Reservoir size** (`self_join`, n = 2·10⁴, 8·10⁸ events): keeping only 1,000 events,
  δ = 41.5 versus 45.1 with the full 100,000-event reservoir — **same sign**. Heapsort at
  n = 2·10⁴: −19.0 → −22.8, same sign.

**Interpretation.** The sign of δ is robust to sampling — supporting streaming
computation — and nearly robust to `b`. The magnitude and the ≈ 0 band are **properties
of the chosen `b = 10`**: the "near-zero" boundary used in the cross-domain test (and the
H4 decision rule) is not invariant to resolution. Conclusions should rest on the
**sign** of δ with `b` fixed in the protocol, not on its absolute value.

### T4 — The boundary of the signature

**(a) The δ ≈ 0 overlap, quantified.** Radix δ = +3.09 and `hash_probe` δ = −0.07, both at
D ≪ 1, fall in the same region of the `(D, δ)` plane; the lag-1 autocorrelation of the
band means over 64 bands, ρ₁, separates them clearly: **0.821 vs 0.023**.

**(b) A "tent" trajectory** (synthetic; zero net trend, strong structure):

| passes | δ (10 bands) | ρ₁ (64 bands) | ρ₁ (10 bands) |
|---:|---:|---:|---:|
| 1 | +0.007 | **0.994** | 0.743 |
| 2 | +0.007 | **0.977** | 0.194 |
| 4 | +0.007 | **0.913** | −0.852 |

A trajectory with zero net trend by construction has δ ≈ 0 — the linear fit cannot
represent it — while ρ₁ detects its structure (0.99). Two caveats follow: ρ₁ computed
over few bands is unreliable (−0.85 at 10 bands for the 4-pass tent), so a temporal-
structure descriptor requires high resolution (M ≥ 64); and the δ ≈ 0 region is not
noise but the boundary of what a two-feature signature can resolve. Characterizing that
region is the subject of a companion study in preparation.

## Summary

| Objection | Holds? | Evidence |
|---|---|---|
| The banded estimator is incidental | **Partially** | trivial estimators reproduce 15/15 verdicts; `d_raw` ρ = 0.997 |
| δ does not measure order | **No** | T2: δ collapses 145–378× under permutation |
| Heapsort separation is explained by any trend statistic | **Yes, more strongly** | order-blind μ separates with the same effect size (2.05σ) |
| Thresholds are circular / data-calibrated | Partially | verdicts have wide margins; the boundary itself is not stressed |
| Applicability to production observability is not demonstrated | **Yes** | framing, not result |
| Narrow empirical base | **Yes** | no real production workloads; no negative cases |
| Structural blind spot | **Yes, quantified** | T4: tent δ = 0.007, ρ₁ = 0.99 |
| Robustness to `b` / reservoir size | **Mixed** | sign robust; magnitude and ≈ 0 band depend on `b` |

**Overall.** δ is supported as an **order-dependent feature** (T2, and H1/H2, where μ
fails) that is robust to sampling (T3). Three claims must be stated more narrowly than in
earlier drafts: the Heapsort separation is also achieved by an order-blind statistic; the
banded estimator is one of many equivalent trend estimators; and the magnitude of δ and
the ≈ 0 band depend on the number of bands.

## Limits of this harness

The corpus reuses the workloads of the main study (no real production workloads); the
tent trajectory is synthetic and not instrumented through the formal pipeline; μ is the
normalized mean accessed index — a moment of the *index* histogram, distinct from the
*temporal* density histogram discussed in the methodology. The purpose is adversarial
(to locate weaknesses), not to benchmark the method.
