"""
Deduplication in two layers, cheapest first.

Layer 1 - exact ID dedup: O(n). Removes tweets the scraper genuinely
saw twice (e.g. a tweet stayed on screen across two scroll passes).

Layer 2 - near-duplicate dedup: catches copy-pasted / bot-repeated
tweets that have the *same wording* but a different tweet_id (because
username or timestamp differ). Uses a normalized-text fingerprint
(hash of lowercased, punctuation-stripped content) so it's still O(n)
- not O(n^2) - and is safe to run on the full dataset.

A third, fuzzier layer (catching tweets that are *almost* the same but
not identical, e.g. "Nifty is up 200 pts!!" vs "Nifty is up 200 pts")
is included as an optional TF-IDF + cosine-similarity pass, but is only
run on reasonably sized batches (see `MAX_ROWS_FOR_FUZZY_PASS` below)
because it is O(n^2) in the worst case. For the "10x more data"
scalability requirement, the right fix is to replace this pass with
MinHash + Locality-Sensitive Hashing (e.g. the `datasketch` library),
which turns the same idea into an approximate O(n) operation - noted
in decision.md rather than implemented here, to keep this codebase's
dependency list small for a take-home assignment.
"""
import hashlib
import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)

_NON_ALNUM_RE = re.compile(r"[^\w\s]", re.UNICODE)
MAX_ROWS_FOR_FUZZY_PASS = 5000


def _fingerprint(text: str) -> str:
    normalized = _NON_ALNUM_RE.sub("", text.lower())
    normalized = re.sub(r"\s+", "", normalized)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["tweet_id"], keep="first")
    logger.info("Exact-ID dedup: removed %d rows", before - len(df))
    return df


def near_duplicate_dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    before = len(df)
    df = df.copy()
    df["_fingerprint"] = df["content"].apply(_fingerprint)
    df = df.drop_duplicates(subset=["_fingerprint"], keep="first")
    df = df.drop(columns=["_fingerprint"])
    logger.info("Near-duplicate (fingerprint) dedup: removed %d rows", before - len(df))
    return df


def fuzzy_dedup(df: pd.DataFrame, threshold: float = 0.92) -> pd.DataFrame:
    """Cosine-similarity pass on TF-IDF vectors of the remaining tweets.

    Skipped automatically above MAX_ROWS_FOR_FUZZY_PASS rows - see the
    module docstring for why, and how this would be replaced for a
    larger dataset.
    """
    if df.empty or len(df) < 2:
        return df
    if len(df) > MAX_ROWS_FOR_FUZZY_PASS:
        logger.info(
            "Skipping fuzzy dedup pass: %d rows exceeds the %d-row safe limit for an O(n^2) "
            "similarity comparison. See dedup.py docstring for the MinHash/LSH alternative.",
            len(df), MAX_ROWS_FOR_FUZZY_PASS,
        )
        return df

    vectorizer = TfidfVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(df["content"].tolist())
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0)

    to_drop = set()
    n = similarity.shape[0]
    for i in range(n):
        if i in to_drop:
            continue
        # Any row j>i that is near-identical to i is treated as a
        # duplicate of i and dropped, keeping the earlier one.
        dupes = np.where(similarity[i, i + 1:] >= threshold)[0] + (i + 1)
        to_drop.update(dupes.tolist())

    before = len(df)
    df = df.drop(df.index[list(to_drop)])
    logger.info("Fuzzy (TF-IDF cosine) dedup: removed %d rows", before - len(df))
    return df


def deduplicate(df: pd.DataFrame, fuzzy_threshold: float = 0.92) -> Tuple[pd.DataFrame, dict]:
    """Run all dedup layers and return the cleaned df plus a small report."""
    start_count = len(df)
    df = exact_dedup(df)
    after_exact = len(df)
    df = near_duplicate_dedup(df)
    after_fingerprint = len(df)
    df = fuzzy_dedup(df, threshold=fuzzy_threshold).reset_index(drop=True)
    after_fuzzy = len(df)

    report = {
        "start_count": start_count,
        "removed_exact_id": start_count - after_exact,
        "removed_fingerprint": after_exact - after_fingerprint,
        "removed_fuzzy": after_fingerprint - after_fuzzy,
        "final_count": after_fuzzy,
    }
    return df, report
