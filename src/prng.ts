/**
 * Deterministic pseudo-random number generator (mulberry32).
 *
 * Required for reproducibility: with a fixed seed, the whole chain
 * (input -> trace -> signature) is reproducible bit for bit.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Returns a uniform integer in [min, max). */
export function intBetween(rng: () => number, min: number, max: number): number {
  return min + Math.floor(rng() * (max - min));
}
