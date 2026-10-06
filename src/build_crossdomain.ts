/**
 * Cross-domain validation driver for the (D, δ) signature.
 *
 * Tests the claim that (D, δ) characterizes FAMILIES of access patterns rather than
 * individual algorithms, by measuring four non-sorting workloads (`src/workloads.ts`)
 * under the SAME thresholds and the SAME density reference as the sorting atlas:
 *   - E(ref) = events of merge/aleatorio at the same n (recomputed with the same seed,
 *     hence identical to the atlas).
 *   - Decision boundaries: inflated load at D ≈ 5; drift sign at δ = 0.
 * No threshold is recalibrated: this is a TRANSFER test. The hypotheses and numerical
 * predictions are stated in `docs/preregistration.md`.
 *
 * Usage:
 *   node src/build_crossdomain.ts [--sizes 1000,2000,5000,10000,20000]
 *                                 [--base-seed 20260731] [--max-stored 100000]
 *                                 [--quad-cap 20000] [--out output/crossdomain.json]
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { mulberry32 } from "./prng.ts";
import { generateInput } from "./inputs.ts";
import { SignalStore } from "./signal.ts";
import { ALGORITHMS } from "./sorts.ts";
import { indexTrajectoryDrift, relativeDensity } from "./signature.ts";
import { WORKLOADS, QUADRATIC_WORKLOADS, type WorkloadName } from "./workloads.ts";

const D_INFLATED = 5; // inflated-load boundary, calibrated on the sorting atlas

function eventBound(name: string, n: number): number {
  // safe upper bound on events per workload (≥ actual count), used to size the internal histogram.
  switch (name) {
    case "self_join": return 2 * n * n + 4 * n;
    case "tree_reduce": return 2 * n + 4;
    default: return n + 4; // linear_reduce, hash_probe
  }
}

/** E(ref) = events of merge/aleatorio at the same n (identical to the sorting atlas). */
function refEvents(n: number, seed: number, maxStored: number): number {
  const input = generateInput(n, "aleatorio", seed);
  const store = new SignalStore(maxStored, mulberry32(seed ^ 0x9e3779b9), 0, n * Math.ceil(Math.log2(Math.max(2, n))) * 6);
  ALGORITHMS.merge.sort(input.slice(), store);
  return store.signal.originalLength;
}

interface Row {
  workload: string; family: string; hypothesis: string; n: number;
  events: number; D: number; drift: number; refEvents: number; sampled: boolean;
  regime: "inflada" | "normal"; driftSign: "+" | "-" | "≈0";
}

function classifyRegime(D: number): "inflada" | "normal" {
  return D > D_INFLATED ? "inflada" : "normal";
}
function classifyDrift(delta: number): "+" | "-" | "≈0" {
  if (Math.abs(delta) < 10) return "≈0";
  return delta > 0 ? "+" : "-";
}

function main(): void {
  const argv = process.argv.slice(2);
  const get = (f: string) => { const i = argv.indexOf(f); return i >= 0 ? argv[i + 1] : undefined; };
  const sizes = (get("--sizes") ?? "1000,2000,5000,10000,20000").split(",").map(Number);
  const baseSeed = Number(get("--base-seed") ?? 20260731);
  const maxStored = Number(get("--max-stored") ?? 100000);
  const quadCap = Number(get("--quad-cap") ?? 20000);
  const out = get("--out") ?? "output/crossdomain.json";
  const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

  const refByN: Record<number, number> = {};
  for (const n of sizes) refByN[n] = refEvents(n, baseSeed, maxStored);

  const rows: Row[] = [];
  const names = Object.keys(WORKLOADS) as WorkloadName[];

  for (const name of names) {
    const wl = WORKLOADS[name];
    for (const n of sizes) {
      if (QUADRATIC_WORKLOADS.has(name) && n > quadCap) continue;
      // Random content; non-sorting workloads are content-independent (the access pattern
      // is determined by the algorithm). Execution is deterministic (fixed seed), so a
      // single run yields the exact trace.
      const input = generateInput(n, "aleatorio", baseSeed);
      const store = new SignalStore(maxStored, mulberry32(baseSeed ^ 0x9e3779b9), 0, eventBound(name, n));
      const result = wl.run(input.slice(), store);
      if (!wl.verify(input, result)) throw new Error(`workload postcondition failed: ${name}/n=${n} (result=${result})`);
      const s = store.signal;
      const events = s.originalLength;
      const drift = indexTrajectoryDrift(s.signal, s.positions, n);
      const D = relativeDensity(events, refByN[n]);
      rows.push({
        workload: name, family: wl.family, hypothesis: wl.hypothesis, n,
        events, D: Number(D.toFixed(4)), drift: Number(drift.toFixed(3)),
        refEvents: refByN[n], sampled: s.sampled,
        regime: classifyRegime(D), driftSign: classifyDrift(drift),
      });
      process.stdout.write(
        `${name.padEnd(14)} ${wl.family.padEnd(18)} n=${String(n).padStart(6)}  ` +
        `E=${String(events).padStart(11)}  D=${D.toExponential(2)}  δ=${drift.toFixed(1)}‰  ` +
        `[${classifyRegime(D)}, δ${classifyDrift(drift)}]\n`,
      );
    }
  }

  // Pre-registered hypotheses, evaluated at the largest measured n of each workload.
  const lastByWorkload = (name: string) => rows.filter((r) => r.workload === name).at(-1)!;
  const verdicts = {
    H1: (() => { const r = lastByWorkload("linear_reduce"); return { pass: r.drift > 40 && r.D < 1, drift: r.drift, D: r.D }; })(),
    H2: (() => { const r = lastByWorkload("tree_reduce"); return { pass: r.drift < 0, drift: r.drift, D: r.D }; })(),
    H3: (() => { const r = lastByWorkload("self_join"); return { pass: r.D > D_INFLATED, drift: r.drift, D: r.D }; })(),
    H4: (() => { const r = lastByWorkload("hash_probe"); return { pass: Math.abs(r.drift) < 10, drift: r.drift, D: r.D }; })(),
  };

  const payload = {
    generatedAt: new Date().toISOString(),
    method: "cross-domain transfer of (D, δ) — non-sort workloads under sort-calibrated thresholds",
    densityReference: "merge/aleatorio at the same n (E_ref), identical to atlas",
    thresholds: { D_inflated: D_INFLATED, drift_zero_band: 10 },
    params: { sizes, baseSeed, maxStored, quadCap },
    preregistration: "docs/preregistration.md",
    verdicts,
    rows,
  };
  const outPath = path.resolve(ROOT, out);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));

  console.log("\n── Pre-registered hypotheses ──");
  for (const [h, v] of Object.entries(verdicts)) {
    console.log(`  ${h}: ${v.pass ? "SUPPORTED" : "NOT SUPPORTED"}  (D=${v.D}, δ=${v.drift}‰)`);
  }
  console.log(`\nOK — ${rows.length} rows written to ${outPath}`);
}

main();
