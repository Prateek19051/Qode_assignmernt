# Indian Market Tweet Intelligence Pipeline

A data collection and analysis pipeline that turns tweets about Indian
stock markets (`#nifty50`, `#sensex`, `#intraday`, `#banknifty`) into a
composite trading signal, built for the Qode Advisors take-home
assignment.

**Before you read the code, please read `decision.md`.** It explains,
in plain English, the main engineering decisions and trade-offs — this
README covers *how to run it*, decision.md covers *why it's built this
way*.

## ⚠️ Known limitation - please read first

This was built in a sandboxed environment with **no internet access**,
so two things could not be tested end-to-end there:

1. **Live scraping against x.com** - the Selenium scraper (`src/scraper/`)
   is complete, real code, but was never run against the live site in
   that sandbox (no network, no real browser session). See
   `decision.md` section 1 for exactly what this means and how the
   pipeline still gets tested end-to-end anyway (using a synthetic
   dataset that mimics the scraper's real output).
2. **Writing/reading Parquet** - `pyarrow` could not be `pip install`-ed
   offline, so `src/processing/storage.py` (which writes Parquet) was
   not run in that sandbox either. The rest of the pipeline was
   verified with a CSV stand-in. This is a completely standard library
   with no unusual setup — the moment you run `pip install -r
   requirements.txt` on a normal machine, it will work.

**Before you submit this**, please run it yourself on a machine with
internet access:
```
pip install -r requirements.txt
python main.py --mode synthetic
```
and ideally also try `--mode live` for a short while (see "Running the
real scraper" below) so you can honestly say you saw it pull live
tweets. This matters both for correctness and for your interview - you
should be able to speak to what you actually watched run.

## Project structure
```
qode-assignment/
├── main.py                      # pipeline entry point
├── config.yaml                  # all tunable settings in one place
├── requirements.txt
├── decision.md                  # plain-English write-up of design decisions
├── src/
│   ├── scraper/
│   │   ├── twitter_scraper.py   # Selenium scraper for x.com search pages
│   │   └── selectors.py         # CSS selectors, kept separate since X changes its HTML often
│   ├── processing/
│   │   ├── cleaner.py           # text/unicode cleaning, type coercion
│   │   ├── dedup.py             # 3-layer deduplication
│   │   └── storage.py           # partitioned Parquet read/write
│   ├── analysis/
│   │   ├── feature_engineering.py  # TF-IDF + custom market-lexicon features
│   │   ├── signals.py              # composite signal + bootstrap confidence interval
│   │   └── visualization.py        # memory-efficient (streaming/sampled) plots
│   ├── utils/
│   │   ├── logger.py
│   │   └── config.py
│   └── generate_sample_data.py  # synthetic data generator (see decision.md §1)
├── tests/                       # unit tests for cleaner, dedup, signals
└── data/
    ├── raw/                     # raw scraped/synthetic tweets (JSON)
    ├── processed/               # cleaned, deduplicated Parquet dataset
    └── sample_output/           # composite signal CSV, plots, run summary
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Selenium needs a Chrome/Chromium binary installed on your machine.
`webdriver-manager` (in requirements.txt) downloads the matching
chromedriver automatically — you don't need to install that part
yourself.

## Running it

### Quick demo (no browser, no internet-scraping needed)
Uses the synthetic data generator so you can see the whole pipeline —
cleaning, dedup, Parquet storage, TF-IDF, signal aggregation, plots —
run in under a minute:
```bash
python main.py --mode synthetic --n-tweets 2200
```
Output:
- `data/raw/latest_raw.json` — raw tweet records
- `data/processed/` — cleaned + deduplicated Parquet dataset, partitioned by date
- `data/sample_output/composite_signal.csv` — hourly composite signal with confidence intervals
- `data/sample_output/signal_over_time.png`, `engagement_distribution.png`
- `data/sample_output/run_summary.json` — counts at each pipeline stage

### Running the real scraper
```bash
python main.py --mode live
```
This opens one headless Chrome window per hashtag (see
`config.yaml → scraper.max_concurrent_workers`) and scrolls each
hashtag's live search page, collecting tweets until the login wall
appears or the per-hashtag target is reached. Read the docstring at
the top of `src/scraper/twitter_scraper.py` before running this — it
explains the login-wall limit and the Terms-of-Service considerations.

### Running tests
```bash
pytest tests/ -v
```

## Configuration
Everything tunable lives in `config.yaml` — hashtags, scrape delays,
Parquet partitioning, TF-IDF size, bootstrap iterations, etc. Change
values there rather than editing source files.


## Approach summary
1. **Collect**: Selenium scrapes public X search pages for each
   hashtag (no login, no paid API), with randomized delays and a
   login-wall detector so it stops cleanly rather than looping forever.
2. **Clean**: Unicode-normalize text (NFC), strip URLs, coerce types,
   extract cashtags, drop unusable rows, with every drop logged.
3. **Deduplicate**: three layers, cheapest first — exact tweet-ID match,
   then a normalized-text fingerprint (catches copy-pasted duplicates),
   then a bounded TF-IDF cosine-similarity pass for near-duplicates.
4. **Store**: partitioned Parquet, partitioned by date, so a later
   query for "just today" doesn't scan the whole history.
5. **Analyze**: TF-IDF vectors plus a small hand-built bullish/bearish
   market lexicon (English + Hinglish), combined into an
   engagement-weighted composite signal per hour, with a bootstrap
   confidence interval so low-volume hours are visibly less certain.
6. **Visualize**: plots are built from streamed batches and reservoir
   sampling rather than loading the full dataset into memory, so the
   same code works whether there are 2,000 or 200,000 rows.
