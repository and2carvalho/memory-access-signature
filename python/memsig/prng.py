"""Python port of src/prng.ts — deterministic mulberry32 PRNG.

Reproduces the JavaScript behaviour bit for bit. The JavaScript mulberry32 relies on
signed 32-bit integer arithmetic through `Math.imul`, `|0`, `>>>` and `^`; these
operators are emulated in Python with 32-bit masking and `to_int32` (truncation to a
signed 32-bit integer).

Original JavaScript (src/prng.ts):
    let a = seed >>> 0;
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;

Reference: https://github.com/bryc/code/blob/master/jshash/PRNGs.md
"""

MASK32 = 0xFFFFFFFF


def to_int32(x: int) -> int:
    """Truncates x to a signed 32-bit integer (equivalent to `| 0`)."""
    x &= MASK32
    return x - 0x100000000 if x & 0x80000000 else x


def imul(a: int, b: int) -> int:
    """Math.imul — signed 32-bit multiplication (low 32 bits of the product)."""
    return to_int32(to_int32(a) * to_int32(b))


def ushr(a: int, n: int) -> int:
    """>>> — unsigned 32-bit right shift."""
    return (to_int32(a) & MASK32) >> n


def mulberry32(seed: int):
    """Returns a generator equivalent to mulberry32(seed) in src/prng.ts."""
    a = seed & MASK32  # `let a = seed >>> 0;`

    def next_():
        nonlocal a
        a = to_int32(a + 0x6D2B79F5)  # (a + 0x6d2b79f5) | 0
        t = to_int32(a ^ ushr(a, 15))  # a ^ (a >>> 15)
        t = imul(t, 1 | a)  # Math.imul(a ^ (a>>>15), 1|a)
        # (t + Math.imul(t ^ (t>>>7), 61|t)) ^ t
        t = to_int32((t + imul(t ^ ushr(t, 7), 61 | t)) ^ t)
        # ((t ^ (t>>>14)) >>> 0) / 4294967296
        return (to_int32(t ^ ushr(t, 14)) & MASK32) / 4294967296

    return next_


def int_between(rng, lo: int, hi: int) -> int:
    """intBetween — uniform integer in [lo, hi)."""
    return lo + int(rng() * (hi - lo))
