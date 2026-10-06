# Pre-registration — Cross-domain validation of the (D, δ) signature

> **Translation note.** This is an English translation of the original pre-registration,
> written in Portuguese and reproduced unchanged in
> [`preregistration.pt-BR.md`](preregistration.pt-BR.md). The Portuguese text is the
> authoritative version. The hypotheses, predictions and decision criteria below are
> translated literally; only file paths were updated to the layout of this repository.
>
> **Editorial note on temporal precedence (added 2026-10-06).** The original text states
> that it precedes the result file `output/crossdomain.json` "in time and in the
> repository history". The second part cannot be verified: the pre-registration, the
> driver and the results were committed together, in a single commit of the original
> (private) research repository on 2026-08-03. That the hypotheses were fixed before the
> measurement is therefore a statement by the author, not independent evidence. See
> *Threats to validity* in [`cross-domain-results.md`](cross-domain-results.md).

**Written BEFORE running `src/build_crossdomain.ts`.** This document fixes the hypotheses
and the numerical predictions *before* any measurement of the non-sorting workloads, in
order to rule out post-hoc interpretation.

## Claim under test

The `(D, δ)` signature does not recognize *algorithms*; it identifies **fundamental
families of memory-access patterns**, of which a sorting algorithm is only one instance.
If this holds, workloads **outside sorting** that instantiate the same family should fall
in the **same region** of the `(log D, δ)` plane — using the **same thresholds**
calibrated on the sorting corpus (inflated-load boundary D ≈ 5; sign boundary δ = 0) and
the **same density reference** `E(ref)` = Merge Sort on random input at the same n.

No threshold is re-tuned for the new domain. This is a **transfer** test.

## The four families and their non-sorting instances

| Family | Sorting instance (existing corpus) | Non-sorting instance (this test) |
|---|---|---|
| Forward scan (δ > 0) | Merge, Radix | `linear_reduce` (linear scan / reduction) |
| Root return (δ < 0) | Heapsort | `tree_reduce` (bottom-up fold, leaves → root) |
| Random access (δ ≈ 0) | *none* | `hash_probe` (random probes) |
| Quadratic overload (D ≫ 5) | Insertion, naive QuickSort | `self_join` (nested-loop O(n²)) |

## Pre-registered hypotheses

**H1 — Forward scan ⟹ δ > 0.**
Forward sweeps produce a positive drift.
*Prediction:* `linear_reduce` has **δ > +40‰** and **D ≪ 1** (light O(n) load).

**H2 — Root return ⟹ δ < 0.**
Traversals that return towards the origin/root produce a negative drift.
*Prediction:* `tree_reduce` has **δ < 0** (the sign is the discriminator, not the
magnitude).

**H3 — Quadratic workloads ⟹ D > 5, regardless of the algorithmic family.**
The inflated-load boundary detects quadratic degeneration/explosion even *outside*
sorting and even when the O(n²) cost is **structural** (not triggered by adversarial
data).
*Prediction:* `self_join` has **D > 5** (expected D ~ 10²–10³ at n = 10⁴). No prediction
on δ is registered for H3: the "quadratic overload" family is defined by D, not by δ.

**H4 — Random probing ⟹ δ ≈ 0.**
Random access without a temporal trend produces a drift close to zero.
*Prediction:* `hash_probe` has **|δ| < 10‰** and **D ≪ 1**.

## Decision criteria (fixed a priori)

- H1 is supported if δ(linear_reduce) > +40 **and** D < 1.
- H2 is supported if δ(tree_reduce) < 0.
- H3 is supported if D(self_join) > 5.
- H4 is supported if |δ(hash_probe)| < 10.

A **falsified** hypothesis is a scientific result, not a failure: it raises the question
of which access structure explains the deviation. The outcome (supported × falsified)
will be reported as measured, without retroactive adjustment of the predictions.

## Measurement protocol

- Sizes n ∈ {10³, 2·10³, 5·10³, 10⁴, 2·10⁴}; `self_join` (O(n²)) bounded by `--quad-cap`.
- Array content: `aleatorio` (random) pattern, base seed `20260731` (same seed family as
  the sorting atlas). Non-sorting workloads are, in general, **content-independent** (the
  access pattern is determined by the algorithm, not by the data) — noted as a difference
  with respect to the sorting algorithms.
- `E(ref)` = events of Merge Sort on random input at the same n (recomputed with the same
  seed, hence identical to the atlas). `D = E(workload)/E(ref)`;
  `δ = indexTrajectoryDrift(...)` from `src/signature.ts`.
- Consolidated output in `output/crossdomain.json`; no raw trace is stored.

_Author: André Cunha Antero de Carvalho (PPGCO/UFU)._
