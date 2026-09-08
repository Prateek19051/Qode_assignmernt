import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.cleaner import clean_dataframe


def _tweet(**overrides):
    base = {
        "tweet_id": "u|2024-01-01T00:00:00|hello",
        "username": "u",
        "content": "Nifty is looking bullish today #nifty50 https://example.com/x",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "likes": 5,
        "retweets": 1,
        "replies": 0,
        "hashtags": ["nifty50"],
        "mentions": [],
        "source_hashtag": "nifty50",
        "scraped_at": "2024-01-01T00:05:00+00:00",
    }
    base.update(overrides)
    return base


def test_strips_urls_from_content():
    df = clean_dataframe([_tweet()])
    assert "https://" not in df.loc[0, "content"]


def test_drops_empty_content_rows():
    df = clean_dataframe([_tweet(content="   ")])
    assert len(df) == 0


def test_drops_unparseable_timestamp():
    df = clean_dataframe([_tweet(timestamp="not-a-date")])
    assert len(df) == 0


def test_extracts_cashtags():
    df = clean_dataframe([_tweet(content="Watch $NIFTY and $BANKNIFTY today")])
    assert set(df.loc[0, "cashtags"]) == {"NIFTY", "BANKNIFTY"}


def test_negative_engagement_clipped_to_zero():
    df = clean_dataframe([_tweet(likes=-5)])
    assert df.loc[0, "likes"] == 0


def test_empty_input_returns_empty_df_with_columns():
    df = clean_dataframe([])
    assert len(df) == 0
    assert "content" in df.columns
