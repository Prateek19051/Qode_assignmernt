"""
Entry point that runs the full pipeline end to end:

    scrape (or load synthetic data) -> clean -> deduplicate -> store as
    Parquet -> engineer features -> aggregate into a trading signal ->
    produce memory-efficient plots -> print a summary report.

Usage:
    python main.py --mode synthetic      # default, always works, no network needed
    python main.py --mode live           # runs the real Selenium scraper against x.com

See decision.md section 1 for why --mode synthetic is the default.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.processing.cleaner import clean_dataframe
from src.processing.dedup import deduplicate
from src.processing.storage import write_parquet, read_parquet, read_parquet_batches
from src.analysis.feature_engineering import add_engineered_features, build_tfidf_features
from src.analysis.signals import aggregate_signals
from src.analysis.visualization import (
    streaming_time_series,
    reservoir_sample,
    plot_signal_over_time,
    plot_engagement_distribution,
)

logger = get_logger(__name__)


def run_live_scrape(config: dict) -> list:
    """Scrape all configured hashtags concurrently, one browser per
    hashtag, using a thread pool. Returns a flat list of tweet dicts.
    """
    from src.scraper.twitter_scraper import TwitterScraper  # imported lazily: selenium
                                                              # + a real chromedriver are
                                                              # only needed for this path

    scraper_cfg = config["scraper"]
    hashtags = scraper_cfg["hashtags"]
    per_hashtag_target = max(1, scraper_cfg["target_tweet_count"] // len(hashtags))
    max_workers = scraper_cfg.get("max_concurrent_workers", 2)

    all_tweets = []

    def _scrape_one(hashtag: str) -> list:
        scraper = TwitterScraper(config)
        scraper.start()
        try:
            tweets = scraper.scrape_hashtag(hashtag, per_hashtag_target)
            return [t.to_dict() for t in tweets]
        finally:
            scraper.close()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scrape_one, tag): tag for tag in hashtags}
        for future in as_completed(futures):
            tag = futures[future]
            try:
                result = future.result()
                logger.info("Hashtag #%s finished with %d tweets", tag, len(result))
                all_tweets.extend(result)
            except Exception:
                logger.exception("Scraping #%s failed", tag)

    return all_tweets


def run_synthetic(n_tweets: int) -> list:
    from src.generate_sample_data import generate

    logger.info("Generating %d synthetic tweets (see decision.md section 1 for why)", n_tweets)
    return generate(n_tweets=n_tweets)


def main():
    parser = argparse.ArgumentParser(description="Indian market tweet intelligence pipeline")
    parser.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    parser.add_argument("--n-tweets", type=int, default=2200, help="only used in synthetic mode")
    args = parser.parse_args()

    config = load_config()
    start_time = time.time()

    # ---- 1. Collection -------------------------------------------------
    if args.mode == "live":
        raw_tweets = run_live_scrape(config)
    else:
        raw_tweets = run_synthetic(args.n_tweets)

    if not raw_tweets:
        logger.error("No tweets collected - stopping.")
        return

    os.makedirs(config["storage"]["raw_dir"], exist_ok=True)
    raw_path = os.path.join(config["storage"]["raw_dir"], "latest_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_tweets, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d raw tweets to %s", len(raw_tweets), raw_path)

    # ---- 2. Clean --------------------------------------------------------
    df = clean_dataframe(raw_tweets)
    logger.info("Cleaned dataset: %d rows", len(df))

    # ---- 3. Deduplicate ----------------------------------------------------
    df, dedup_report = deduplicate(df, fuzzy_threshold=config["processing"]["dedup_similarity_threshold"])
    logger.info("Dedup report: %s", dedup_report)

    # ---- 4. Store as Parquet ------------------------------------------------
    write_parquet(df, config["storage"]["processed_dir"], partition_col=config["storage"]["partition_column"])

    # ---- 5. Feature engineering ------------------------------------------
    df = add_engineered_features(df)
    tfidf_matrix = build_tfidf_features(
        df["content"].tolist(),
        max_features=config["analysis"]["tfidf_max_features"],
        ngram_range=config["analysis"]["tfidf_ngram_range"],
    )
    logger.info("TF-IDF feature matrix shape: %s", tfidf_matrix.shape)

    # ---- 6. Signal aggregation with confidence intervals -----------------
    signal_df = aggregate_signals(
        df,
        iterations=config["analysis"]["bootstrap_iterations"],
        confidence_level=config["analysis"]["confidence_level"],
    )

    os.makedirs("data/sample_output", exist_ok=True)
    signal_out_path = "data/sample_output/composite_signal.csv"
    signal_df.to_csv(signal_out_path, index=False)
    logger.info("Saved composite signal table to %s", signal_out_path)

    # ---- 7. Memory-efficient visualization --------------------------------
    # Re-read from the Parquet dataset in batches, even though `df` is
    # already in memory here, specifically to exercise and demonstrate
    # the streaming/batched code path described in visualization.py -
    # this is the code path that would be used on a dataset too large to
    # hold in memory at all.
    batches_for_ts = read_parquet_batches(config["storage"]["processed_dir"], batch_size=500)
    engagement_series = streaming_time_series(batches_for_ts, value_col="likes", freq="1h")

    plot_signal_over_time(
        signal_df.set_index("bucket")["composite_signal"] if not signal_df.empty else engagement_series,
        output_path="data/sample_output/signal_over_time.png",
    )

    batches_for_sample = read_parquet_batches(config["storage"]["processed_dir"], batch_size=500)
    sample = reservoir_sample(batches_for_sample, column="engagement_raw" if "engagement_raw" in df.columns else "likes", sample_size=2000)
    plot_engagement_distribution(sample, output_path="data/sample_output/engagement_distribution.png")

    # ---- 8. Summary --------------------------------------------------------
    elapsed = time.time() - start_time
    summary = {
        "mode": args.mode,
        "raw_tweet_count": len(raw_tweets),
        "after_cleaning": dedup_report["start_count"],
        "after_dedup": dedup_report["final_count"],
        "duplicates_removed": dedup_report["start_count"] - dedup_report["final_count"],
        "time_buckets_with_signal": len(signal_df),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open("data/sample_output/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
