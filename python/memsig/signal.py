"""Python port of src/signal.ts — bounded-memory trace collector.

Uses reservoir sampling that is uniform in time: once the number of events exceeds
`max_stored`, a fixed-size uniform sample is kept together with the original relative
ORDER of events (required by the drift δ). The `sampled` flag records whether sampling
occurred; `positions` holds the original order of each kept event.
"""


class SignalStore:
    def __init__(self, max_stored: int = 100_000, rng=__import__("random").random,
                 histogram_buckets: int = 0, expected_length: int = 0):
        self._keep: list[int] = []
        self._keep_pos: list[int] = []
        self._count = 0
        self._rng = rng
        self._max_stored = max_stored
        self._sampling_active = False
        self._hist_buckets = histogram_buckets
        self._expected_length = max(histogram_buckets, expected_length)
        self._hist = [0] * histogram_buckets if histogram_buckets > 0 else []

    def push(self, v: int) -> None:
        t = self._count + 1
        self._count = t
        if self._hist_buckets > 0:
            b = min(self._hist_buckets - 1,
                    int(((t - 1) * self._hist_buckets) / self._expected_length))
            self._hist[b] += 1
        if len(self._keep) < self._max_stored:
            self._keep.append(v)
            self._keep_pos.append(t)
            return
        self._sampling_active = True
        r = int(self._rng() * t)
        if r < self._max_stored:
            self._keep[r] = v
            self._keep_pos[r] = t

    @property
    def signal(self):
        return {
            "signal": self._keep,
            "positions": self._keep_pos if self._sampling_active else None,
            "originalLength": self._count,
            "sampled": self._sampling_active,
            "histogram": self._hist,
            "histogramBuckets": self._hist_buckets,
        }
