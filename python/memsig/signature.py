"""Python port of src/signature.ts — the (D, δ) signature of a trace.

D — RELATIVE ACCESS DENSITY = E(a) / E(ref)
δ — INDEX-TRAJECTORY DRIFT: slope (‰ per 1% of the trace) of the mean accessed index
    over time, estimated by linear regression over `bands` temporal bands.
"""

import math


def index_trajectory_drift(signal, positions, n: int, bands: int = 10) -> float:
    """Index-trajectory drift — mirrors indexTrajectoryDrift in src/signature.ts."""
    if len(signal) < 8:
        return 0.0
    pairs = sorted(
        (positions[i] if positions is not None else i, v) for i, v in enumerate(signal)
    )
    L = len(pairs)
    means: list[float] = []
    for b in range(bands):
        # IMPORTANT: mirror Math.floor((b/bands)*L) with JavaScript float arithmetic.
        # `(b * L) // bands` diverges whenever (b/bands)*L is not exact in floating
        # point (e.g. 0.7*45000 = 31499.999999999996 → floor 31499, not 31500).
        lo = math.floor((b / bands) * L)
        hi = math.floor(((b + 1) / bands) * L)
        s = 0.0
        for i in range(lo, hi):
            s += pairs[i][1]
        means.append(s / max(1, hi - lo))
    y_max = max(1, n - 1)
    m = len(means)
    sx = sy = sxx = sxy = 0.0
    for i in range(m):
        x = i / (m - 1)          # normalized temporal position, 0..1
        y = means[i] / y_max     # normalized index, 0..1
        sx += x; sy += y; sxx += x * x; sxy += x * y
    denom = m * sxx - sx * sx
    slope = (m * sxy - sx * sy) / (denom if denom else 1e-12)
    return (1000 * slope) / bands


def relative_density(events: int, ref_events: int) -> float:
    """Relative access density D = events(a) / events(reference)."""
    return events / max(1, ref_events)
