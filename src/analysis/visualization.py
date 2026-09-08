"""
Memory-efficient plotting for datasets that may not fit comfortably in
RAM (the "10x more data" scalability requirement).

Two techniques used, matching the two different kinds of plot needed:

1. Streaming aggregation (for the signal-over-time chart): the data is
   read in batches (see storage.read_parquet_batches), and each batch
   only contributes to a small set of running totals (sum + count per
   time bucket). The raw rows are discarded after each batch, so peak
   memory is roughly "one batch" no matter how many batches there are,
   not "the whole dataset".

2. Reservoir sampling (for the engagement-distribution chart): when you
   need a scatter/histogram of individual tweets rather than an
   aggregate, you cannot keep a running sum - you need actual example
   rows. Reservoir sampling lets you read a stream of unknown/large
   length once and end up with a uniform random sample of a fixed size
   (e.g. 5,000 points), without ever holding more than that many rows
   in memory at once.
"""
import random
from typing import Iterator, Optional

import matplotlib
matplotlib.use("Agg")  # no display available / needed - write straight to file
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def streaming_time_series(
    batches: Iterator[pd.DataFrame], value_col: str, time_col: str = "timestamp", freq: str = "1h"
) -> pd.Series:
    """Compute a mean-per-time-bucket series across many small batches
    instead of concatenating them all into one DataFrame first."""
    sums: dict = {}
    counts: dict = {}

    for batch in batches:
        if batch.empty:
            continue
        bucketed = batch[time_col].dt.floor(freq)
        grouped = batch.groupby(bucketed)[value_col]
        for bucket, total in grouped.sum().items():
            sums[bucket] = sums.get(bucket, 0.0) + total
        for bucket, n in grouped.count().items():
            counts[bucket] = counts.get(bucket, 0) + n

    if not sums:
        return pd.Series(dtype=float)

    buckets = sorted(sums.keys())
    means = {b: sums[b] / counts[b] for b in buckets}
    return pd.Series(means).sort_index()


def reservoir_sample(
    batches: Iterator[pd.DataFrame], column: str, sample_size: int = 5000, seed: int = 42
) -> list:
    """Classic Algorithm R reservoir sampling over a stream of batches.

    Guarantees a uniform random sample of `sample_size` values from the
    full stream, seen or not seen in advance, while only ever holding
    `sample_size` values in memory.
    """
    rng = random.Random(seed)
    reservoir: list = []
    seen = 0

    for batch in batches:
        for value in batch[column].tolist():
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(value)
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_size:
                    reservoir[j] = value

    logger.info("Reservoir-sampled %d of %d total values seen", len(reservoir), seen)
    return reservoir


def plot_signal_over_time(series: pd.Series, output_path: str, title: str = "Composite signal over time") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not series.empty:
        ax.plot(series.index, series.values, marker="o", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Composite signal")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)


def plot_engagement_distribution(
    sample: list, output_path: str, title: str = "Engagement distribution (sampled)"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if sample:
        ax.hist(sample, bins=40)
    ax.set_title(f"{title} (n={len(sample)})")
    ax.set_xlabel("Engagement score (log1p scale)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)
