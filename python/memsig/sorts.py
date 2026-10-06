"""Python port of src/sorts.ts — instrumented sorting algorithms.

Trace-capture contract for the main array A:
  - every read of A in a comparison records the position read;
  - every write to A (shift, copy or swap) records the position written;
  - indices are always global (0..n-1) and can be normalized by (n-1);
  - auxiliary buffers (e.g. Merge Sort's `tmp`) are separate memory and are excluded
    from the trace — a deliberate scope limitation.

The exact order of `signal(...)` calls follows the TypeScript implementation, since it
determines E(a) and δ.
"""

from typing import Callable

SignalSink = Callable[[int], None]


def insertion_sort(a: list[int], signal: SignalSink) -> None:
    n = len(a)
    for i in range(1, n):
        current = a[i]
        j = i - 1
        while j >= 0 and a[j] > current:
            signal(j)      # read of A[j] in the comparison
            signal(j + 1)  # write A[j+1] = A[j] (shift)
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = current


def merge_sort(a: list[int], signal: SignalSink) -> None:
    n = len(a)
    tmp = [0] * n

    def merge(lo: int, mid: int, hi: int) -> None:
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            signal(i)  # read of the left-run element
            signal(j)  # read of the right-run element
            if a[i] <= a[j]:
                tmp[k] = a[i]; i += 1
            else:
                tmp[k] = a[j]; j += 1
            k += 1
        while i < mid:
            signal(i)
            tmp[k] = a[i]; i += 1; k += 1
        while j < hi:
            signal(j)
            tmp[k] = a[j]; j += 1; k += 1
        for p in range(lo, hi):
            signal(p)  # write back to A
            a[p] = tmp[p]

    def sort(lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        mid = (lo + hi) >> 1
        sort(lo, mid)
        sort(mid, hi)
        merge(lo, mid, hi)

    sort(0, n)


def quick_sort(a: list[int], signal: SignalSink) -> None:
    def swap(x: int, y: int) -> None:
        if x == y:
            return
        signal(x)
        signal(y)
        a[x], a[y] = a[y], a[x]

    def median_of_three(lo: int, hi: int) -> int:
        mid = (lo + hi) >> 1
        if a[lo] > a[mid]:
            swap(lo, mid)
        if a[lo] > a[hi]:
            swap(lo, hi)
        if a[mid] > a[hi]:
            swap(mid, hi)
        return mid

    def partition(lo: int, hi: int) -> int:
        p = median_of_three(lo, hi)
        swap(p, lo)
        pivot = a[lo]
        i = lo
        for j in range(lo + 1, hi + 1):
            signal(j)  # read in the comparison against the pivot
            if a[j] < pivot:
                i += 1
                swap(i, j)
        swap(lo, i)
        return i

    def qsort(lo: int, hi: int) -> None:
        if lo >= hi:
            return
        p = partition(lo, hi)
        qsort(lo, p - 1)
        qsort(p + 1, hi)

    qsort(0, len(a) - 1)


def heap_sort(a: list[int], signal: SignalSink) -> None:
    n = len(a)

    def swap(x: int, y: int) -> None:
        if x == y:
            return
        signal(x)
        signal(y)
        a[x], a[y] = a[y], a[x]

    def sift_down(start: int, size: int) -> None:
        i = start
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            largest = i
            if l < size:
                signal(l)
                if a[l] > a[largest]:
                    largest = l
            if r < size:
                signal(r)
                if a[r] > a[largest]:
                    largest = r
            if largest == i:
                break
            swap(i, largest)
            i = largest

    for i in range((n >> 1) - 1, -1, -1):
        sift_down(i, n)
    for i in range(n - 1, 0, -1):
        swap(0, i)
        sift_down(0, i)


def quick_sort_naive(a: list[int], signal: SignalSink) -> None:
    n = len(a)

    def swap(x: int, y: int) -> None:
        if x == y:
            return
        signal(x)
        signal(y)
        a[x], a[y] = a[y], a[x]

    def partition(lo: int, hi: int) -> int:
        pivot = a[hi]  # pivot = last element → degenerates on sorted/reversed input
        i = lo
        for j in range(lo, hi):
            signal(j)  # read in the comparison against the pivot
            if a[j] < pivot:
                swap(i, j)
                i += 1
        swap(i, hi)
        return i

    stack = [0, n - 1]
    while stack:
        hi = stack.pop()
        lo = stack.pop()
        if lo >= hi:
            continue
        p = partition(lo, hi)
        stack.append(p + 1)
        stack.append(hi)
        stack.append(lo)
        stack.append(p - 1)


def shell_sort(a: list[int], signal: SignalSink) -> None:
    n = len(a)
    h = 1
    while h < n / 3:
        h = 3 * h + 1

    while h >= 1:
        for i in range(h, n):
            current = a[i]
            j = i
            while j >= h and a[j - h] > current:
                signal(j - h)  # read of A[j-h] in the comparison
                signal(j)      # write A[j] = A[j-h] (shift)
                a[j] = a[j - h]
                j -= h
            a[j] = current
        h = h // 3


def radix_sort(a: list[int], signal: SignalSink) -> None:
    n = len(a)
    if n <= 1:
        return
    max_val = max(a) if a else 0

    out = [0] * n
    count = [0] * 256
    exp = 1

    while exp <= max_val:
        count[:] = [0] * 256
        # Pass 1 — count per digit (sequential read of A).
        for i in range(n):
            signal(i)  # read of A[i]
            count[(a[i] // exp) % 256 & 255] += 1
        # Prefix sums over count.
        for i in range(1, 256):
            count[i] += count[i - 1]
        # Pass 2 — distribute into `out`.
        for i in range(n - 1, -1, -1):
            d = (a[i] // exp) % 256 & 255
            signal(i)  # read of A[i]
            count[d] -= 1
            out[count[d]] = a[i]
        # Pass 3 — copy back into A.
        for i in range(n):
            signal(i)  # write of A[i]
            a[i] = out[i]
        exp *= 256


ALGORITHMS = {
    "insertion": insertion_sort,
    "merge": merge_sort,
    "quicksort": quick_sort,
    "quicksort_naive": quick_sort_naive,
    "heapsort": heap_sort,
    "shellsort": shell_sort,
    "radix": radix_sort,
}
