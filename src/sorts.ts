/**
 * Instrumented sorting algorithms.
 *
 * Trace-capture contract for the main array A:
 *  - every read of A in a comparison records the position read;
 *  - every write to A (shift, copy or swap) records the position written;
 *  - indices are always global (0..n-1) and can be normalized by (n-1);
 *  - auxiliary buffers (e.g. Merge Sort's `tmp`) are separate memory and are excluded
 *    from the trace — a deliberate scope limitation.
 */

export interface SignalSink {
  push(v: number): void;
}

/** Insertion Sort — incremental; worst case O(n²). */
export function insertionSort(a: number[], signal: SignalSink): void {
  const n = a.length;
  for (let i = 1; i < n; i++) {
    const current = a[i];
    let j = i - 1;
    while (j >= 0 && a[j] > current) {
      signal.push(j);     // read of A[j] in the comparison
      signal.push(j + 1); // write A[j+1] = A[j] (shift)
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = current;
  }
}

/** Top-down Merge Sort (with an auxiliary buffer) — O(n log n). */
export function mergeSort(a: number[], signal: SignalSink): void {
  const n = a.length;
  const tmp = new Array<number>(n);

  function merge(lo: number, mid: number, hi: number): void {
    let i = lo;
    let j = mid;
    let k = lo;
    while (i < mid && j < hi) {
      signal.push(i); // read of the left-run element (global index)
      signal.push(j); // read of the right-run element (global index)
      if (a[i] <= a[j]) tmp[k++] = a[i++];
      else tmp[k++] = a[j++];
    }
    while (i < mid) {
      signal.push(i);
      tmp[k++] = a[i++];
    }
    while (j < hi) {
      signal.push(j);
      tmp[k++] = a[j++];
    }
    for (let p = lo; p < hi; p++) {
      signal.push(p); // write back to A (global index)
      a[p] = tmp[p];
    }
  }

  function sort(lo: number, hi: number): void {
    if (hi - lo <= 1) return;
    const mid = (lo + hi) >> 1;
    sort(lo, mid);
    sort(mid, hi);
    merge(lo, mid, hi);
  }

  sort(0, n);
}

/** QuickSort — Lomuto partition with median-of-three pivot (avoids O(n²) on reversed input). */
export function quickSort(a: number[], signal: SignalSink): void {
  function swap(x: number, y: number): void {
    if (x === y) return;
    signal.push(x);
    signal.push(y);
    const t = a[x];
    a[x] = a[y];
    a[y] = t;
  }

  function medianOfThree(lo: number, hi: number): number {
    const mid = (lo + hi) >> 1;
    if (a[lo] > a[mid]) swap(lo, mid);
    if (a[lo] > a[hi]) swap(lo, hi);
    if (a[mid] > a[hi]) swap(mid, hi);
    return mid;
  }

  function partition(lo: number, hi: number): number {
    const p = medianOfThree(lo, hi);
    swap(p, lo);
    const pivot = a[lo];
    let i = lo;
    for (let j = lo + 1; j <= hi; j++) {
      signal.push(j); // read in the comparison against the pivot
      if (a[j] < pivot) {
        i++;
        swap(i, j);
      }
    }
    swap(lo, i);
    return i;
  }

  function qsort(lo: number, hi: number): void {
    if (lo >= hi) return;
    const p = partition(lo, hi);
    qsort(lo, p - 1);
    qsort(p + 1, hi);
  }

  qsort(0, a.length - 1);
}

/** HeapSort — O(n log n) regardless of input. */
export function heapSort(a: number[], signal: SignalSink): void {
  const n = a.length;

  function swap(x: number, y: number): void {
    if (x === y) return;
    signal.push(x);
    signal.push(y);
    const t = a[x];
    a[x] = a[y];
    a[y] = t;
  }

  function siftDown(start: number, size: number): void {
    let i = start;
    for (;;) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let largest = i;
      if (l < size) {
        signal.push(l);
        if (a[l] > a[largest]) largest = l;
      }
      if (r < size) {
        signal.push(r);
        if (a[r] > a[largest]) largest = r;
      }
      if (largest === i) break;
      swap(i, largest);
      i = largest;
    }
  }

  for (let i = (n >> 1) - 1; i >= 0; i--) siftDown(i, n);
  for (let i = n - 1; i > 0; i--) {
    swap(0, i);
    siftDown(0, i);
  }
}

/**
 * Naive QuickSort — Lomuto partition with the last element as pivot (CLRS).
 *
 * On sorted or reverse-sorted input the pivot is always an extreme value, yielding
 * unbalanced (n-1 | 0) partitions and degeneration to O(n²) — the classical worst case
 * that the median-of-three variant (`quickSort`) avoids. Implemented iteratively (explicit
 * stack) to avoid call-stack overflow in the worst case; the sequence of comparisons and
 * swaps is identical to the recursive formulation.
 */
export function quickSortNaive(a: number[], signal: SignalSink): void {
  const n = a.length;

  function swap(x: number, y: number): void {
    if (x === y) return;
    signal.push(x);
    signal.push(y);
    const t = a[x];
    a[x] = a[y];
    a[y] = t;
  }

  function partition(lo: number, hi: number): number {
    const pivot = a[hi]; // pivot = last element → degenerates on sorted/reversed input
    let i = lo;
    for (let j = lo; j < hi; j++) {
      signal.push(j); // read in the comparison against the pivot
      if (a[j] < pivot) {
        swap(i, j);
        i++;
      }
    }
    swap(i, hi);
    return i;
  }

  const stack: number[] = [0, n - 1];
  while (stack.length > 0) {
    const hi = stack.pop() as number;
    const lo = stack.pop() as number;
    if (lo >= hi) continue;
    const p = partition(lo, hi);
    // Push the RIGHT side first so that the LEFT side is processed first (LIFO),
    // matching the processing order of the recursive version (qsort(lo, p-1) first).
    stack.push(p + 1, hi);
    stack.push(lo, p - 1);
  }
}

/**
 * Shellsort — Knuth gap sequence (h = 3h+1, then h = ⌊h/3⌋).
 *
 * Memory-access strategy: strided accesses with large gaps that shrink gradually to 1;
 * the expected signature is a POSITIVE drift with modulation (each gap reduction pulls
 * the scan pointer back across successive passes).
 */
export function shellSort(a: number[], signal: SignalSink): void {
  const n = a.length;
  let h = 1;
  while (h < n / 3) h = 3 * h + 1; // largest Knuth gap < n

  while (h >= 1) {
    for (let i = h; i < n; i++) {
      const current = a[i];
      let j = i;
      while (j >= h && a[j - h] > current) {
        signal.push(j - h); // read of A[j-h] in the comparison
        signal.push(j);     // write A[j] = A[j-h] (shift)
        a[j] = a[j - h];
        j -= h;
      }
      a[j] = current;
    }
    h = Math.floor(h / 3);
  }
}

/**
 * LSD Radix Sort — base 256 (8-bit digits).
 *
 * Memory-access strategy: sequential bucket-by-bucket sweeps (one counting sort per
 * digit) with high locality and no jumps. Because every pass restarts at index 0, the
 * net trend over the whole trace is close to zero (δ ≈ 0).
 */
export function radixSort(a: number[], signal: SignalSink): void {
  const n = a.length;
  if (n <= 1) return;
  let maxVal = 0;
  for (let i = 0; i < n; i++) if (a[i] > maxVal) maxVal = a[i];

  const out = new Array<number>(n);
  const count = new Int32Array(256);
  let exp = 1;

  while (exp <= maxVal) {
    count.fill(0);
    // Pass 1 — count per digit (sequential read of A).
    for (let i = 0; i < n; i++) {
      signal.push(i); // read of A[i]
      count[(a[i] / exp) % 256 & 255]++;
    }
    // Prefix sums over count.
    for (let i = 1; i < 256; i++) count[i] += count[i - 1];
    // Pass 2 — distribute into `out` (sequential read, output positions).
    for (let i = n - 1; i >= 0; i--) {
      const d = (a[i] / exp) % 256 & 255;
      signal.push(i); // read of A[i]
      out[--count[d]] = a[i];
    }
    // Pass 3 — copy back into A (sequential write).
    for (let i = 0; i < n; i++) {
      signal.push(i); // write of A[i]
      a[i] = out[i];
    }
    exp *= 256;
  }
}

/** Expected complexity class per algorithm (experiment label: lento = quadratic, rapido = sub-quadratic). */
export type ComplexityLabel = "lento" | "rapido";

export const ALGORITHMS = {
  insertion: { sort: insertionSort, complexity: "lento" as ComplexityLabel },
  merge: { sort: mergeSort, complexity: "rapido" as ComplexityLabel },
  quicksort: { sort: quickSort, complexity: "rapido" as ComplexityLabel },
  quicksort_naive: { sort: quickSortNaive, complexity: "rapido" as ComplexityLabel },
  heapsort: { sort: heapSort, complexity: "rapido" as ComplexityLabel },
  shellsort: { sort: shellSort, complexity: "rapido" as ComplexityLabel },
  radix: { sort: radixSort, complexity: "rapido" as ComplexityLabel },
} as const;

export type AlgorithmName = keyof typeof ALGORITHMS;
