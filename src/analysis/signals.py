"""
Combines per-tweet features into one composite trading signal per time
bucket, with a confidence interval around it.

Why a confidence interval at all: a signal built from 5 tweets in an
hour is much less trustworthy than one built from 500 tweets, even if
both average out to the same score. A single number hides that.
Reporting a confidence interval lets whoever consumes this signal
(e.g. a downstream trading strategy) automatically down-weight
low-volume, low-confidence hours instead of treating every hour's
signal as equally reliable.

Method: bootstrap resampling. For each time bucket, repeatedly resample
(with replacement) the tweets in that bucket, recompute the weighted
average signal each time, and take the 2.5th/97.5th percentile of the
resulting distribution as a 95% confidence interval. This is used
instead of a plain standard-error formula because the per-tweet signal
is not normally distributed (it's bounded in [-1, 1] and often spiky),
and bootstrapping makes no assumption about the underlying distribution.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _weighted_composite(scores: np.ndarray, weights: np.ndarray) -> float:
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float(np.sum(scores * weights) / total_weight)


def bootstrap_confidence_interval(
    scores: np.ndarray,
    weights: np.ndarray,
    iterations: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Return (lower, upper) bounds of the bootstrap confidence interval."""
    if len(scores) == 0:
        return (0.0, 0.0)
    if len(scores) == 1:
        # A single observation has no spread to bootstrap - report the
        # point value as both bounds rather than a fabricated interval.
        return (float(scores[0]), float(scores[0]))

    rng = np.random.default_rng(random_state)
    n = len(scores)
    resampled_means = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = _weighted_composite(scores[idx], weights[idx])

    alpha = 1 - confidence_level
    lower = float(np.percentile(resampled_means, 100 * (alpha / 2)))
    upper = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def aggregate_signals(
    df: pd.DataFrame,
    freq: str = "1h",
    iterations: int = 500,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Bucket tweets by time and produce one composite signal row per
    bucket: weighted-average sentiment (weighted by engagement), the
    tweet volume behind it, and a bootstrap confidence interval.

    Expects `df` to already have `lexicon_score` and `engagement_score`
    columns (see feature_engineering.add_engineered_features).
    """
    if df.empty:
        return pd.DataFrame(
            columns=["bucket", "composite_signal", "ci_lower", "ci_upper", "tweet_count"]
        )

    df = df.copy()
    df["bucket"] = df["timestamp"].dt.floor(freq)

    rows = []
    for bucket, group in df.groupby("bucket"):
        scores = group["lexicon_score"].to_numpy()
        # Weight = 1 + engagement_score, so a tweet with zero engagement
        # still counts (weight 1) instead of being erased from the
        # average entirely.
        weights = 1.0 + group["engagement_score"].to_numpy()

        composite = _weighted_composite(scores, weights)
        ci_lower, ci_upper = bootstrap_confidence_interval(
            scores, weights, iterations=iterations, confidence_level=confidence_level
        )
        rows.append(
            {
                "bucket": bucket,
                "composite_signal": composite,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "tweet_count": len(group),
            }
        )

    result = pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
    logger.info("Aggregated %d time buckets from %d tweets", len(result), len(df))
    return result
