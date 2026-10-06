"""Python port of src/inputs.ts — input generators.

Three input patterns (identifiers kept as in the published data):
  - "aleatorio"       (random):        uniform distribution.
  - "quase_ordenado"  (nearly sorted): ascending order with ~5% local swaps.
  - "inverso"         (reversed):      strictly descending order.
"""

from .prng import mulberry32

PATTERNS = ("aleatorio", "quase_ordenado", "inverso")


def generate_input(n: int, pattern: str, seed: int) -> list[int]:
    rng = mulberry32(seed)

    if pattern == "aleatorio":
        return [int(rng() * 1_000_000) for _ in range(n)]

    if pattern == "inverso":
        return [n - i for i in range(n)]

    # quase_ordenado: ascending order + ~5% local swaps.
    arr = list(range(n))
    swaps = max(1, int(n * 0.05))
    for _ in range(swaps):
        p = int(rng() * n)
        off = 1 + int(rng() * 5)
        q = p + off if p + off < n else p - off
        if q < 0 or q >= n:
            continue
        arr[p], arr[q] = arr[q], arr[p]
    return arr
