/**
 * Bounded-memory trace collector.
 *
 * The execution trace of Insertion Sort in its worst case has O(n²) events
 * (e.g. n = 10,000 -> ~50 million shifts), which is impractical to store in full.
 * To keep the pipeline scalable and reproducible, the collector uses *reservoir
 * sampling*: once the number of events exceeds `maxStored`, it keeps a uniform random
 * sample of fixed size together with the original position of each kept event.
 *
 * The reservoir is uniform in time: it preserves the relative order of events (required
 * by the drift δ) at the cost of fine temporal resolution. The `sampled` flag records
 * whether sampling took place; `positions` holds the original order of each sampled event.
 *
 * Optionally, the collector maintains a temporal density histogram (count per bucket,
 * O(1) per event) when `histogramBuckets > 0`. The signature driver (`build_atlas.ts`)
 * does not use it; it is provided for pipelines that need temporal density without
 * storing every event.
 */
export interface StoredSignal {
  readonly signal: readonly number[];
  /** Original position (order) of each event in the stream, when sampled. */
  readonly positions: readonly number[] | null;
  readonly originalLength: number;
  readonly sampled: boolean;
  /** Temporal density histogram (count per bucket), O(1) per event. */
  readonly histogram: readonly number[] | Uint32Array;
  readonly histogramBuckets: number;
}

export class SignalStore {
  private readonly keep: number[] = [];
  private readonly keepPos: number[] = [];
  private count = 0;
  private readonly rng: () => number;
  private readonly maxStored: number;
  private samplingActive = false;
  private readonly hist: Uint32Array;
  private readonly histBuckets: number;
  private readonly expectedLength: number;

  /**
   * @param maxStored         events kept by the reservoir (shorter traces are not sampled).
   * @param rng               deterministic PRNG driving the reservoir.
   * @param histogramBuckets  resolution of the density histogram (0 disables it).
   * @param expectedLength    safe upper bound on the total number of events (≥ originalLength).
   */
  constructor(
    maxStored = 100_000,
    rng: () => number = Math.random,
    histogramBuckets = 0,
    expectedLength = 0,
  ) {
    this.maxStored = maxStored;
    this.rng = rng;
    this.histBuckets = histogramBuckets;
    this.expectedLength = Math.max(histogramBuckets, expectedLength);
    this.hist = histogramBuckets > 0 ? new Uint32Array(histogramBuckets) : new Uint32Array(0);
  }

  /** Records one event (the global index of the accessed memory position). */
  push(v: number): void {
    const t = ++this.count;
    if (this.histBuckets > 0) {
      const b = Math.min(this.histBuckets - 1, Math.floor(((t - 1) * this.histBuckets) / this.expectedLength));
      this.hist[b]++;
    }
    if (this.keep.length < this.maxStored) {
      this.keep.push(v);
      this.keepPos.push(t);
      return;
    }
    if (!this.samplingActive) this.samplingActive = true;
    const r = Math.floor(this.rng() * t);
    if (r < this.maxStored) {
      this.keep[r] = v;
      this.keepPos[r] = t;
    }
  }

  get signal(): StoredSignal {
    return {
      signal: this.keep,
      // Positions are exposed only when sampling occurred; for complete traces the
      // position of element k equals its array index and need not be stored.
      positions: this.samplingActive ? this.keepPos : null,
      originalLength: this.count,
      sampled: this.samplingActive,
      histogram: this.hist,
      histogramBuckets: this.histBuckets,
    };
  }
}
