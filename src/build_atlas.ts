/**
 * Atlas driver for the (D, δ) signature.
 *
 * Sweeps the full experimental grid — algorithms × input patterns × problem sizes ×
 * trials — to characterize how (D, δ) scale with n and to assess the cross-n stability
 * of the decision boundaries (δ = 0 and the inflated-load boundary on D).
 *
 * For each (algorithm × pattern × n × trial) the driver generates the input from a fixed
 * seed, runs the instrumented algorithm with order-preserving reservoir sampling, and
 * computes (D, δ) in memory. Only the consolidated JSON is written; raw traces are not
 * stored because they are reproducible bit for bit from the fixed seeds.
 *
 * Density reference: E(ref) = events of merge/aleatorio AT THE SAME n, so that D is
 * comparable across sizes (merge/aleatorio ≡ 1.0 at every n).
 *
 * Usage:
 *   node src/build_atlas.ts [--sizes 1000,2000,5000,10000,20000]
 *                            [--trials 5] [--base-seed 20260731]
 *                            [--max-stored 100000] [--out output/atlas.json]
 *   # Quadratic algorithms (insertion, quicksort_naive) are skipped above --quad-cap
 *   # (default 20000) to keep wall-clock time tractable.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { mulberry32 } from "./prng.ts";
import { generateInput, PATTERNS, type Pattern } from "./inputs.ts";
import { SignalStore } from "./signal.ts";
import { ALGORITHMS, type AlgorithmName } from "./sorts.ts";
import { indexTrajectoryDrift, relativeDensity } from "./signature.ts";

const HISTOGRAM_BUCKETS = 0; // (D, δ) does not require the density histogram
const QUADRATIC = new Set<AlgorithmName>(["insertion", "quicksort_naive"]);

function expectedLengthBound(algo: AlgorithmName, n: number): number {
  const log2n = Math.ceil(Math.log2(Math.max(2, n)));
  let bound: number;
  switch (algo) {
    case "insertion": bound = n * n; break;
    case "quicksort_naive": bound = n * n; break;
    case "merge": bound = n * log2n * 4; break;
    case "quicksort": bound = n * log2n * 3; break;
    case "heapsort": bound = n * log2n * 4; break;
    case "shellsort": bound = n * log2n * log2n * 4; break;
    case "radix": bound = n * 16; break;
  }
  return Math.ceil(bound * 1.5);
}

function isSorted(a: readonly number[]): boolean {
  for (let i = 1; i < a.length; i++) if (a[i - 1] > a[i]) return false;
  return true;
}

interface RunResult { events: number; drift: number; timeMs: number; sampled: boolean; }

function runOnce(algo: AlgorithmName, pattern: Pattern, n: number, seed: number, maxStored: number): RunResult {
  const input = generateInput(n, pattern, seed);
  const store = new SignalStore(maxStored, mulberry32(seed ^ 0x9e3779b9), HISTOGRAM_BUCKETS, expectedLengthBound(algo, n));
  const a = input.slice();
  const t0 = performance.now();
  ALGORITHMS[algo].sort(a, store);
  const timeMs = performance.now() - t0;
  if (!isSorted(a)) throw new Error(`sort postcondition failed: ${algo}/${pattern}/n=${n}/seed=${seed}`);
  const s = store.signal;
  const drift = indexTrajectoryDrift(s.signal, s.positions, n);
  return { events: s.originalLength, drift, timeMs, sampled: s.sampled };
}

function mean(xs: number[]): number { return xs.reduce((a, b) => a + b, 0) / xs.length; }

function parseArgs(argv: string[]) {
  const get = (f: string) => { const i = argv.indexOf(f); return i >= 0 ? argv[i + 1] : undefined; };
  return {
    sizes: (get("--sizes") ?? "1000,2000,5000,10000,20000").split(",").map(Number),
    trials: Number(get("--trials") ?? 5),
    baseSeed: Number(get("--base-seed") ?? 20260731),
    maxStored: Number(get("--max-stored") ?? 100000),
    quadCap: Number(get("--quad-cap") ?? 20000),
    out: get("--out") ?? "output/atlas.json",
  };
}

function main(): void {
  const { sizes, trials, baseSeed, maxStored, quadCap, out } = parseArgs(process.argv.slice(2));
  const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const algos = Object.keys(ALGORITHMS) as AlgorithmName[];

  // Global density reference per n: events of merge/aleatorio.
  const refEventsByN: Record<number, number> = {};
  for (const n of sizes) refEventsByN[n] = runOnce("merge", "aleatorio", n, baseSeed, maxStored).events;

  interface Row {
    algo: string; pattern: string; n: number;
    events: number; D: number; drift: number;
    refEvents: number; sampled: boolean;
    trials: { events: number; drift: number }[];
  }
  const rows: Row[] = [];

  for (const algo of algos) {
    for (const pattern of PATTERNS) {
      for (const n of sizes) {
        if (QUADRATIC.has(algo) && n > quadCap) continue; // O(n²) wall-time guard
        const per = [];
        for (let t = 0; t < trials; t++) {
          per.push(runOnce(algo, pattern, n, baseSeed + t, maxStored));
        }
        const events = Math.round(mean(per.map((r) => r.events)));
        const drift = mean(per.map((r) => r.drift));
        const D = relativeDensity(events, refEventsByN[n]);
        rows.push({
          algo, pattern, n, events,
          D: Number(D.toFixed(4)),
          drift: Number(drift.toFixed(3)),
          refEvents: refEventsByN[n],
          sampled: per[0].sampled,
          trials: per.map((r) => ({ events: r.events, drift: Number(r.drift.toFixed(3)) })),
        });
        process.stdout.write(
          `${algo.padEnd(16)} ${pattern.padEnd(15)} n=${String(n).padStart(6)}  ` +
          `E=${String(events).padStart(11)}  D=${D.toExponential(2)}  δ=${drift.toFixed(1)}‰\n`,
        );
      }
    }
  }

  const payload = {
    generatedAt: new Date().toISOString(),
    method: "signature (D = relative access density, δ = index-trajectory drift); no resonant filter",
    densityReference: "merge/aleatorio at the same n (E_ref)",
    params: { sizes, trials, baseSeed, maxStored, quadCap },
    rows,
  };
  const outPath = path.resolve(ROOT, out);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));
  console.log(`\nOK — ${rows.length} rows written to ${outPath}`);
}

main();
