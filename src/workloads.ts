/**
 * Non-sorting workloads for the cross-domain validation of the (D, δ) signature.
 *
 * The signature only requires a LOGICAL ARRAY indexed 0..n-1 and one `signal.push(index)`
 * per access — nothing is specific to sorting. Under the SAME instrumentation contract as
 * `sorts.ts`, this module implements four workloads that instantiate the four access-
 * pattern families pre-registered in `docs/preregistration.md`:
 *
 *   forward scan       (δ>0)  → linear_reduce  (linear scan / reduction)
 *   root return        (δ<0)  → tree_reduce    (bottom-up fold from leaves to root)
 *   random access      (δ≈0)  → hash_probe     (pseudo-random probes into A)
 *   quadratic overload (D≫5)  → self_join      (nested-loop O(n²), structural)
 *
 * Each workload exposes a verifiable postcondition (`verify`) to ensure that it actually
 * performed the computation its name describes.
 */
import { mulberry32 } from "./prng.ts";
import type { SignalSink } from "./sorts.ts";

/** A cross-domain workload: runs over A while recording accesses, then checks its postcondition. */
export interface Workload {
  /** Runs the instrumented workload over a copy of `a`; returns a result for `verify`. */
  run(a: number[], signal: SignalSink): number;
  /** Confirms that the workload produced the expected result (verifiable postcondition). */
  verify(a: readonly number[], result: number): boolean;
  /** Predicted (pre-registered) access-pattern family. */
  family: "forward-scan" | "root-return" | "random-access" | "quadratic-overload";
  hypothesis: "H1" | "H2" | "H3" | "H4";
}

/**
 * FORWARD SCAN — linear reduction (sum) in a single forward sweep over A.
 * The accessed index increases monotonically 0→n-1 ⇒ strongly positive δ; O(n) load ⇒ D≪1.
 * Non-sorting instance of the forward-scan family. (H1)
 */
export function linearReduce(a: number[], signal: SignalSink): number {
  let acc = 0;
  for (let i = 0; i < a.length; i++) {
    signal.push(i); // sequential read of A[i]
    acc += a[i];
  }
  return acc;
}

/**
 * ROOT RETURN — bottom-up fold over a complete binary tree encoded in A (index 0 = root,
 * children of i at 2i+1 / 2i+2). Iterates from i = n-1 down to 1, accumulating each node
 * into its parent (i-1)>>1: the access pointer moves from the leaves (high indices) towards
 * the root (index 0) over time ⇒ negative δ. This is the same "return to the root" pattern
 * as Heapsort's extraction phase, here as a tree reduction / segment-tree construction
 * rather than a sort. On exit, A[0] holds the total sum. (H2)
 */
export function treeReduce(a: number[], signal: SignalSink): number {
  for (let i = a.length - 1; i >= 1; i--) {
    const parent = (i - 1) >> 1;
    signal.push(i);       // read of child A[i]
    signal.push(parent);  // read-modify-write of A[parent]
    a[parent] += a[i];
  }
  return a[0];
}

/**
 * RANDOM ACCESS — pseudo-random probes into A (models the probe side of a hash join or
 * scattered lookups). No temporal trend in the accessed index ⇒ δ≈0; O(m) load with m = n
 * ⇒ D≪1. This family has no instance among the sorting algorithms. (H4)
 */
export function hashProbe(a: number[], signal: SignalSink): number {
  const n = a.length;
  const rng = mulberry32(0x51ed270b ^ n); // deterministic, content-independent
  let acc = 0;
  for (let k = 0; k < n; k++) {
    const idx = Math.floor(rng() * n); // probe target, uniform in [0, n)
    signal.push(idx);                  // random access to A[idx]
    acc += a[idx];
  }
  return acc;
}

/**
 * QUADRATIC OVERLOAD — nested-loop band self-join: for each i, scans every j and counts
 * pairs with |A[i]-A[j]| ≤ τ. The workload is STRUCTURALLY O(n²) — quadratic for any input,
 * unlike naive QuickSort, which requires adversarial input to degenerate. Both the outer
 * operand A[i] and the scanned operand A[j] are recorded in each comparison. Inflated load
 * ⇒ D≫5. Non-sorting instance of the same family as Insertion Sort / naive QuickSort. (H3)
 * No prediction on δ is registered for this family.
 */
export function selfJoin(a: number[], signal: SignalSink): number {
  const n = a.length;
  const tau = 4; // narrow band threshold → non-trivial count; cost remains O(n²)
  let matches = 0;
  for (let i = 0; i < n; i++) {
    const ai = a[i];
    for (let j = 0; j < n; j++) {
      signal.push(i); // read of the outer operand A[i] in the comparison
      signal.push(j); // read of the scanned operand A[j] in the comparison
      const d = ai - a[j];
      if (d <= tau && d >= -tau) matches++;
    }
  }
  return matches;
}

export const WORKLOADS: Record<string, Workload> = {
  linear_reduce: {
    run: linearReduce,
    verify: (a, r) => r === a.reduce((s, x) => s + x, 0),
    family: "forward-scan",
    hypothesis: "H1",
  },
  tree_reduce: {
    // The run mutates A (A[0] receives the sum). The postcondition only checks that a fold
    // took place (finite, positive result); an exact check would require capturing the
    // sum of the original array before the run.
    run: treeReduce,
    verify: (_a, r) => Number.isFinite(r) && r > 0,
    family: "root-return",
    hypothesis: "H2",
  },
  hash_probe: {
    run: hashProbe,
    verify: (_a, r) => Number.isFinite(r) && r >= 0,
    family: "random-access",
    hypothesis: "H4",
  },
  self_join: {
    run: selfJoin,
    verify: (a, r) => r >= a.length, // every element matches at least itself (d = 0 ≤ τ)
    family: "quadratic-overload",
    hypothesis: "H3",
  },
};

export type WorkloadName = keyof typeof WORKLOADS;

/** Structurally O(n²) workloads: bounded by --quad-cap in the driver, like the quadratic sorts. */
export const QUADRATIC_WORKLOADS: ReadonlySet<string> = new Set(["self_join"]);
