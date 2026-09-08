"""
Storage layer: writes/reads the cleaned tweet dataset as Parquet.

Why Parquet (as the assignment suggests) instead of CSV/JSON:
- Columnar storage means the analysis step (which mostly reads
  `content`, `likes`, `timestamp`) does not have to read the other
  columns off disk at all.
- Built-in compression (snappy by default) shrinks a text-heavy
  dataset noticeably compared to CSV.
- Partitioning by date means a later "just analyze today" query only
  opens today's files instead of scanning the whole history - this is
  what makes the "10x more data" scalability requirement realistic
  without a rewrite: partitioning is what keeps a bigger dataset from
  making every query linearly slower.

Schema (enforced via the dtypes set in cleaner.py before this is
called):
    tweet_id        string
    username        string
    content         string (Unicode, NFC-normalized)
    timestamp       timestamp[UTC]
    likes           int64
    retweets        int64
    replies         int64
    hashtags        list<string>
    mentions        list<string>
    cashtags        list<string>
    source_hashtag  string
    scraped_at      timestamp[UTC]
    date            string   <- partition column
"""
import os
from typing import Iterator, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logger import get_logger

logger = get_logger(__name__)


def write_parquet(df: pd.DataFrame, output_dir: str, partition_col: str = "date") -> None:
    """Append-write the DataFrame into a partitioned Parquet dataset."""
    if df.empty:
        logger.warning("write_parquet called with an empty DataFrame - nothing written")
        return

    os.makedirs(output_dir, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=output_dir,
        partition_cols=[partition_col] if partition_col in df.columns else None,
        # existing_data_behavior="overwrite_or_ignore" would clobber prior
        # runs; we instead let pyarrow add a new file per write so
        # multiple scrape runs on the same day accumulate instead of
        # overwriting each other.
        existing_data_behavior="overwrite_or_ignore",
        basename_template="part-{i}.parquet",
    )
    logger.info("Wrote %d rows to Parquet dataset at %s", len(df), output_dir)


def read_parquet(input_dir: str, columns: Optional[list] = None) -> pd.DataFrame:
    """Read the full partitioned dataset back into memory.

    Only use this for datasets you know fit comfortably in RAM. For the
    "large dataset" / streaming case, use `read_parquet_batches` below
    instead.
    """
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logger.warning("No data found at %s", input_dir)
        return pd.DataFrame()
    return pd.read_parquet(input_dir, columns=columns, engine="pyarrow")


def read_parquet_batches(
    input_dir: str, batch_size: int = 2000, columns: Optional[list] = None
) -> Iterator[pd.DataFrame]:
    """Yield the dataset in fixed-size chunks instead of loading it all at
    once. This is what the "memory-efficient visualization" and
    "10x data" requirements lean on: the visualization and signal
    aggregation code can process one batch, update a running total, and
    discard the batch, so peak memory use stays roughly constant no
    matter how large the underlying dataset grows.
    """
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logger.warning("No data found at %s", input_dir)
        return
    dataset = pq.ParquetDataset(input_dir)
    for fragment in dataset.fragments:
        for batch in fragment.to_table(columns=columns).to_batches(max_chunksize=batch_size):
            yield batch.to_pandas()
