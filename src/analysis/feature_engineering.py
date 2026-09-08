"""
Turns tweet text into numeric features that a trading signal can use.

Two kinds of features, combined:

1. TF-IDF vectors - generic "what words matter in this corpus" features.
   Useful for clustering/classification later, but on their own they
   don't tell you if a word like "crash" is bullish or bearish.

2. A small hand-built market lexicon - domain knowledge about how
   Indian retail traders actually talk (bullish/bearish slang, common
   Hinglish terms). This is what lets the pipeline distinguish
   "Nifty breakout, target 24500" (bullish) from "Nifty crash, exit
   now" (bearish), which pure TF-IDF cannot do by itself.

Both are kept because they answer different questions: TF-IDF answers
"what is this corpus about", the lexicon answers "is the sentiment
positive or negative". A real production system would likely swap the
lexicon for a fine-tuned FinBERT-style model - noted as a next step in
decision.md, not built here, to keep the dependency list to what
scikit-learn already provides.
"""
import re
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# A short, hand-picked lexicon for Indian equity-market chatter.
# Not exhaustive - meant to demonstrate the approach, easy to extend.
BULLISH_TERMS = {
    "buy", "long", "breakout", "rally", "bullish", "upside", "target",
    "support", "gain", "surge", "uptrend", "accumulate", "momentum",
    "tejji", "profit",
}
BEARISH_TERMS = {
    "sell", "short", "breakdown", "crash", "bearish", "downside",
    "resistance", "loss", "plunge", "downtrend", "correction", "mandi",
    "dump", "panic",
}

_WORD_RE = re.compile(r"[a-zA-Z]+")


def lexicon_score(text: str) -> float:
    """Return a score in [-1, 1]: +1 = all bullish words, -1 = all bearish.

    Simple bag-of-words scoring: count bullish vs bearish hits and
    normalize by total hits. A tweet with no matching words scores 0
    (neutral / unknown), not missing - this keeps downstream math simple.
    """
    words = {w.lower() for w in _WORD_RE.findall(text)}
    bull_hits = len(words & BULLISH_TERMS)
    bear_hits = len(words & BEARISH_TERMS)
    total = bull_hits + bear_hits
    if total == 0:
        return 0.0
    return (bull_hits - bear_hits) / total


def build_tfidf_features(
    texts: List[str], max_features: int = 500, ngram_range=(1, 2)
) -> np.ndarray:
    """Fit a TF-IDF vectorizer on the given texts and return the dense
    feature matrix. Kept dense here because downstream steps (bootstrap
    resampling) need row-wise access repeatedly; for a truly large
    corpus this would stay sparse and use `TruncatedSVD` before the
    bootstrap step - noted in decision.md.
    """
    if not texts:
        return np.zeros((0, max_features))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(ngram_range),
        stop_words="english",
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    logger.info(
        "TF-IDF matrix built: %d docs x %d features", matrix.shape[0], matrix.shape[1]
    )
    return matrix.toarray()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lexicon sentiment score, engagement score, and content-length
    features to the DataFrame in place (returns a copy)."""
    df = df.copy()
    df["lexicon_score"] = df["content"].apply(lexicon_score)

    # Engagement is heavily right-skewed (a handful of viral tweets swamp
    # the rest) - log1p compresses that range so one viral tweet doesn't
    # dominate the composite signal below.
    df["engagement_raw"] = df["likes"] + 2 * df["retweets"] + df["replies"]
    df["engagement_score"] = np.log1p(df["engagement_raw"])

    df["content_length"] = df["content"].str.len()
    df["hashtag_count"] = df["hashtags"].apply(len)
    df["cashtag_count"] = df["cashtags"].apply(len)

    return df
