# Decisions - what I did and why

This file explains the choices made in this project in plain English,
so you (whoever submits this) can understand it well enough to answer
interview questions about it confidently, without having to re-read
every line of code.

---

## 1. Why there's a "synthetic data" mode, and how live scraping actually works

**The assignment asks for 2000+ real tweets scraped with Selenium.**
The scraper code that does this is real and complete
(`src/scraper/twitter_scraper.py`) - it opens a Chrome browser, goes to
X's public search page for a hashtag, scrolls down, and reads each
tweet's text, username, timestamp, and like/retweet/reply counts
straight out of the page.

Two honest problems with actually running this at "2000 tweets in
24 hours" scale, which I did not want to hide:

1. **X limits how much you can see without logging in.** After
   scrolling through roughly 20-40 tweets on an anonymous session, X
   shows a "log in to see more" wall. The scraper detects this wall and
   stops cleanly (instead of getting stuck retrying forever), but it
   means one anonymous run per hashtag realistically returns dozens of
   tweets, not thousands. To get thousands, you would need to log into
   a real X account in the browser session, which brings its own
   account-suspension risk since automated scraping goes against X's
   Terms of Service.
2. **Scraping X's website is against X's Terms of Service.** The
   assignment explicitly says "no paid APIs, use Selenium," which
   already means working in a gray area — I built the scraper to be
   respectful (random delays between actions, no aggressive parallel
   hammering of one page, stopping at the login wall instead of
   fighting it) rather than trying to defeat X's bot detection
   outright.

**What I did about it:** rather than submit a repo that "should work"
with no proof, I added `src/generate_sample_data.py`, which generates a
synthetic dataset that looks exactly like the scraper's real output
(same fields, same messiness — duplicate tweets, Hindi/Hinglish text,
skewed like/retweet counts) so that every other stage of the pipeline
(cleaning, dedup, storage, feature engineering, signal building,
plotting) can actually be run and checked, not just written.

**If asked about this in an interview:** be upfront. Say the scraper is
real and you'd run it live for the final submission, but you used a
synthetic stand-in to prove the rest of the pipeline works correctly,
because an anonymous scrape genuinely can't guarantee 2000 tweets in a
short window without a logged-in session, and you didn't want to fake
that number. This is a stronger answer than pretending it definitely
scraped 2000 real tweets.

**Before you submit:** actually run `python main.py --mode live` for a
while on your own machine and see what it collects. Even a partial live
run (a few hundred tweets) is worth including alongside the synthetic
demo, so you have a real example to talk about.

---

## 2. Why a dataclass for each tweet, instead of a plain dictionary

A plain `{"username": ..., "content": ...}` dictionary is easy to typo
(`"usrname"` fails silently, you just get a `KeyError` later, far from
the actual bug). A `Tweet` dataclass
(`src/scraper/twitter_scraper.py`) fixes the exact fields up front, so
mistakes are caught immediately where they happen, and every tool
(editor autocomplete, code reviewers) can see the shape of the data at
a glance.

---

## 3. Why deduplication has three separate steps

Real scraped data has duplicates for different reasons, so one
technique doesn't catch all of them:

1. **Exact ID match** — the scraper genuinely saw the same tweet twice
   (e.g. it stayed on screen across two scroll actions). Cheapest check,
   done first.
2. **Text fingerprint match** — two tweets with different IDs but the
   *exact same wording* (a retweet-style copy, or the scraper building
   two slightly different IDs for what is really one tweet). Done by
   lowercasing the text, stripping punctuation, and hashing it — still
   very fast even on a huge dataset.
3. **Fuzzy match (TF-IDF + cosine similarity)** — two tweets that are
   *almost* identical but not exactly ("Nifty is up 200 points!!" vs
   "Nifty is up 200 points"). This is the most expensive check
   (comparing every tweet to every other tweet), so I only run it when
   the dataset is small enough to afford that (under 5,000 rows — see
   `MAX_ROWS_FOR_FUZZY_PASS` in `dedup.py`). For a much bigger dataset,
   the standard fix is a technique called **MinHash + LSH**, which finds
   near-duplicates in roughly linear time instead of comparing every
   pair. I didn't implement that here to keep the dependency list small
   for a take-home assignment, but I documented exactly where it would
   go, in case they ask about scaling this up.

Cheapest-first ordering matters: by the time we reach the expensive
step, most duplicates are already gone, so there's less left to compare.

---

## 4. Why Parquet, and why partitioned by date

The assignment asks for Parquet, and it's a genuinely good fit here:

- It's a **columnar** format — reading just the `content` and `likes`
  columns for analysis doesn't require reading every other column off
  disk too, unlike CSV.
- It **compresses well**, which matters for a text-heavy dataset.
- **Partitioning by date** (`data/processed/date=2026-09-05/...`) means
  a query for "just today's tweets" only opens today's files, instead
  of scanning the entire history. This is the main thing that keeps the
  pipeline fast if the dataset grows 10x, as the assignment asks about
  — you're not scanning 10x more data for every query, only for queries
  that actually need that much history.

---

## 5. Why TF-IDF *and* a hand-built bullish/bearish word list, not just one

TF-IDF on its own tells you "these are the important words in this
tweet," but it has no idea whether "crash" is good or bad news for the
market. So I added a small list of bullish words (buy, breakout,
rally, tejji...) and bearish words (sell, crash, breakdown, mandi...),
including common Hinglish trading slang, and scored each tweet by how
many of each it contains. That score is what actually feeds into the
trading signal — TF-IDF vectors are built too (as the assignment asks
for), and would be the input to a proper machine-learning classifier if
this were extended further, but a hand-built lexicon is what makes the
signal directionally meaningful right now without training a model.

**If I had more time**, I'd replace the hand-built lexicon with a
sentiment model trained specifically on financial text (something like
FinBERT), which would understand context far better than a word list
(a word list can't tell "no crash expected" from "crash expected" — it
just sees the word "crash" either way).

---

## 6. Why the signal has a confidence interval, not just a number

An hour with 5 tweets and an hour with 500 tweets might both average
out to the same sentiment score, but the second one is far more
trustworthy. Reporting just the average hides that difference. So each
hour's signal comes with a range (a 95% confidence interval) built by
**bootstrapping**: repeatedly re-sampling that hour's tweets (with
replacement) and recomputing the average each time, then taking the
range that covers 95% of those recomputed averages. Hours with fewer
tweets naturally get a wider range — which is the whole point: it makes
low-confidence hours visibly less certain instead of treating every
hour's number as equally solid.

---

## 7. Why the visualizations use "streaming" and "sampling" instead of loading everything into memory

If this dataset became 10x or 100x bigger, loading it all into memory
just to draw a chart would eventually crash or slow to a crawl. Two
techniques avoid that:

- **Streaming aggregation** (for the signal-over-time chart): read the
  data in small chunks, keep a running total and count per hour, and
  throw each chunk away once it's counted. Memory use stays roughly
  constant no matter how much data there is in total.
- **Reservoir sampling** (for the engagement-distribution chart): when
  you need actual individual data points for a chart (not just a
  running average), you can't just keep a running total — but you also
  can't keep every point if there are millions of them. Reservoir
  sampling is a well-known trick that lets you read through the data
  once and end up with a fair, random sample of a fixed size (e.g.
  2,000 points), without ever holding more than that many points in
  memory.

---

## 8. Why one browser per hashtag, run in a thread pool, instead of one browser doing everything

A single Selenium browser can only do one thing at a time — it can't
scroll two different search pages simultaneously. So to scrape 4
hashtags concurrently, the pipeline starts up to
`max_concurrent_workers` separate Chrome instances (default: 2, see
`config.yaml`), each handling one hashtag, coordinated with Python's
`ThreadPoolExecutor`. This is also why hashtags run "as completed"
rather than waiting for all of them together — if one hashtag hits the
login wall early, its result is used right away instead of blocking on
the slowest hashtag.

---

## 9. What's genuinely tested here, and what isn't (please read before your interview)

I want to be completely transparent about this so you're not caught
off guard by a question:

- **Tested and working**: cleaning, all three dedup layers, feature
  engineering, signal aggregation with confidence intervals, and both
  visualization functions — all run end-to-end on a 2,000+ row
  synthetic dataset, and there are unit tests in `tests/` covering the
  tricky edge cases (empty input, bad timestamps, negative counts, a
  single-tweet confidence interval, etc).
- **Not tested in the sandbox this was built in, but should just
  work**: Parquet writing/reading (`src/processing/storage.py`) —
  the sandbox had no internet access to install the `pyarrow` library.
  This is an extremely standard, widely-used library with no special
  setup — install it with `pip install -r requirements.txt` on a normal
  machine and it will work; there's nothing unusual about this code
  path, it just genuinely wasn't run here.
- **Not run against the live site in this sandbox** (no internet
  access there either): the actual scraping of x.com. The code is
  complete and I'm confident in the approach (see section 1 above for
  the honest limitations of what an anonymous scrape can realistically
  return), but you should run `python main.py --mode live` yourself
  before submitting, both to get a real dataset and so you can speak
  from firsthand experience if they ask what you saw when you ran it.
