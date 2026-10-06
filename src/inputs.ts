/**
 * Input generators for the experiments.
 *
 * Three input patterns. Identifiers are kept in Portuguese because they are part of the
 * published data files:
 *  - "aleatorio"       (random):        uniform distribution.
 *  - "quase_ordenado"  (nearly sorted): ascending order with ~5% local swaps.
 *  - "inverso"         (reversed):      strictly descending order.
 */
import { mulberry32 } from "./prng.ts";

export type Pattern = "aleatorio" | "quase_ordenado" | "inverso";

export const PATTERNS: readonly Pattern[] = ["aleatorio", "quase_ordenado", "inverso"];

export function generateInput(
  n: number,
  pattern: Pattern,
  seed: number,
): number[] {
  const rng = mulberry32(seed);

  if (pattern === "aleatorio") {
    const arr = new Array<number>(n);
    for (let i = 0; i < n; i++) arr[i] = Math.floor(rng() * 1_000_000);
    return arr;
  }

  if (pattern === "inverso") {
    const arr = new Array<number>(n);
    for (let i = 0; i < n; i++) arr[i] = n - i;
    return arr;
  }

  // quase_ordenado: ascending order + ~5% local swaps.
  const arr = new Array<number>(n);
  for (let i = 0; i < n; i++) arr[i] = i;
  const swaps = Math.max(1, Math.floor(n * 0.05));
  for (let k = 0; k < swaps; k++) {
    const p = Math.floor(rng() * n);
    const off = 1 + Math.floor(rng() * 5);
    const q = p + off < n ? p + off : p - off;
    if (q < 0 || q >= n) continue;
    const t = arr[p];
    arr[p] = arr[q];
    arr[q] = t;
  }
  return arr;
}
