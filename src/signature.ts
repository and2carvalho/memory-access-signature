/**
 * The (D, δ) signature of an access trace over a logical array.
 *
 * Two low-cost, calibration-free features:
 *
 *   D — RELATIVE ACCESS DENSITY
 *     D(a) = E(a) / E(ref), where E(a) is the total number of accesses performed by
 *     algorithm `a` and E(ref) is that of a fixed global reference (merge sort on random
 *     input at the same n). D is information-equivalent to an access counter; its value is
 *     operational (O(1) state per event, no per-class calibration) rather than algorithmic.
 *     It isolates inflated workload and performance degeneration.
 *
 *   δ — INDEX-TRAJECTORY DRIFT
 *     Slope of the mean accessed position over normalized execution time, estimated by
 *     linear regression over `bands` temporal bands and reported in ‰ per 1% of the trace.
 *     δ summarizes the net direction of the access trajectory (forward scan vs. return
 *     towards the origin), i.e. its low-frequency component — an order-dependent quantity
 *     that global means and marginal histograms do not retain.
 *
 * Both are direct statistics of the trace (a count and a regression slope).
 */

/**
 * Index-trajectory drift: slope (‰ per 1% of the trace) of the mean accessed index over
 * time. `positions` (the original temporal order of each sampled event) is used for
 * ordering; when absent, the array index is taken as the temporal order.
 */
export function indexTrajectoryDrift(
  signal: readonly number[],
  positions: readonly number[] | null,
  n: number,
  bands = 10,
): number {
  if (signal.length < 8) return 0;
  const pairs: Array<[number, number]> = signal.map((v, i) => [positions ? positions[i] : i, v]);
  pairs.sort((a, b) => a[0] - b[0]);
  const L = pairs.length;
  const means: number[] = [];
  for (let b = 0; b < bands; b++) {
    const lo = Math.floor((b / bands) * L);
    const hi = Math.floor(((b + 1) / bands) * L);
    let s = 0;
    for (let i = lo; i < hi; i++) s += pairs[i][1];
    means.push(s / Math.max(1, hi - lo));
  }
  const yMax = Math.max(1, n - 1);
  const m = means.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < m; i++) {
    const x = i / (m - 1); // normalized temporal position, 0..1
    const y = means[i] / yMax; // normalized index, 0..1
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const slope = (m * sxy - sx * sy) / (m * sxx - sx * sx || 1e-12);
  return (1000 * slope) / bands; // ‰ per 1% of the trace
}

/** Relative access density D = events(a) / events(reference). */
export function relativeDensity(events: number, refEvents: number): number {
  return events / Math.max(1, refEvents);
}
