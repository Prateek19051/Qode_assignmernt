"""
Cleans and normalizes raw scraped tweets into a tidy pandas DataFrame.

Design note: this takes a list of dicts (or Tweet.to_dict() output) in,
and returns a DataFrame out. Keeping the scraper's output format
(plain dicts) decoupled from pandas means the scraper module doesn't
need to import pandas at all, and the cleaner can just as easily be
fed dicts that came from a JSON file instead of a live scrape.
"""
import re
import unicodedata
from typing import List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

URL_RE = re.compile(r"https?://\S+")
EXTRA_WHITESPACE_RE = re.compile(r"\s+")
# Common financial tickers/cashtags e.g. $NIFTY, $RELIANCE - kept as a
# separate column since they are a strong, structured trading signal
# and stripping them out of the text would throw that information away.
CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,15})\b")


def _strip_urls(text: str) -> str:
    return URL_RE.sub("", text)


def _extract_cashtags(text: str) -> List[str]:
    return [m.upper() for m in CASHTAG_RE.findall(text)]


def clean_dataframe(records: List[dict]) -> pd.DataFrame:
    """Convert raw tweet dicts into a cleaned, typed DataFrame.

    Steps applied, in order:
    1. Build the DataFrame and drop rows with no text at all.
    2. Normalize Unicode (NFC) so Devanagari/Gujarati text and emoji
       compare equal regardless of how the source encoded them.
    3. Strip URLs out of the display text (kept separately would be a
       nice extension, dropped here since raw links carry no signal
       for the text-to-signal step).
    4. Parse timestamps into real datetime objects; unparseable ones
       become NaT rather than crashing the whole batch.
    5. Coerce engagement counts to non-negative integers.
    6. Extract cashtags ($NIFTY, $RELIANCE, ...) into their own column.
    """
    if not records:
        logger.warning("clean_dataframe called with 0 records")
        return pd.DataFrame(
            columns=[
                "tweet_id", "username", "content", "timestamp", "likes",
                "retweets", "replies", "hashtags", "mentions", "cashtags",
                "source_hashtag", "scraped_at",
            ]
        )

    df = pd.DataFrame.from_records(records)
    before = len(df)
    df = df[df["content"].notna() & (df["content"].str.strip() != "")]
    logger.info("Dropped %d rows with empty content", before - len(df))

    df["content"] = df["content"].apply(
        lambda t: unicodedata.normalize("NFC", str(t))
    )
    df["content"] = df["content"].apply(_strip_urls)
    df["content"] = df["content"].str.replace(EXTRA_WHITESPACE_RE, " ", regex=True).str.strip()

    df["cashtags"] = df["content"].apply(_extract_cashtags)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)

    for col in ("likes", "retweets", "replies"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(int)

    # A tweet with no timestamp can't be used for the "last 24 hours"
    # window filter downstream, so it's better to drop it here with a
    # clear log line than to silently include it or crash later.
    before = len(df)
    df = df[df["timestamp"].notna()]
    logger.info("Dropped %d rows with unparseable timestamp", before - len(df))

    df["date"] = df["timestamp"].dt.date.astype(str)

    return df.reset_index(drop=True)
