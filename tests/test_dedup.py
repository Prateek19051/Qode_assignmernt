import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.processing.dedup import exact_dedup, near_duplicate_dedup, deduplicate


def _df(rows):
    return pd.DataFrame(rows)


def test_exact_dedup_removes_same_tweet_id():
    df = _df([
        {"tweet_id": "a", "content": "hello world"},
        {"tweet_id": "a", "content": "hello world"},
        {"tweet_id": "b", "content": "different"},
    ])
    result = exact_dedup(df)
    assert len(result) == 2


def test_near_duplicate_dedup_catches_punctuation_variants():
    df = _df([
        {"tweet_id": "a", "content": "Nifty is up 200 points!!"},
        {"tweet_id": "b", "content": "nifty is up 200 points"},
        {"tweet_id": "c", "content": "completely unrelated tweet"},
    ])
    result = near_duplicate_dedup(df)
    assert len(result) == 2


def test_deduplicate_report_adds_up():
    df = _df([
        {"tweet_id": "a", "content": "buy nifty now"},
        {"tweet_id": "a", "content": "buy nifty now"},
        {"tweet_id": "b", "content": "sell banknifty now"},
    ])
    result, report = deduplicate(df)
    assert report["start_count"] == 3
    assert report["final_count"] == len(result)
    assert report["final_count"] <= report["start_count"]
