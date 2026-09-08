"""
Selenium-based scraper for public tweets on X (Twitter) search pages.

IMPORTANT - please read before running this against the live site:

1. This scrapes PUBLIC search result pages (no login, no paid API), as the
   assignment requires. X does not offer a free way to do this at scale
   without some form of browser automation, which is why Selenium is used
   here instead of the official API.

2. X actively limits how much an unauthenticated session can see (it shows
   a "login wall" after a couple of dozen tweets). This scraper detects
   that wall and stops cleanly instead of getting stuck. In practice this
   means a single anonymous run will usually return well under the "2000
   tweets in 24 hours" target - see decision.md, section 1, for how this
   assignment handles that gap honestly instead of pretending it isn't
   there.

3. Scraping X's website is against X's Terms of Service. This code is
   written for a technical assignment / educational purpose. If you run
   it, do so on your own account/responsibility, keep request rates low
   (see config.yaml delays), and do not use it to build a product that
   depends on scraping X at scale.

Data structures used here and why:
- A `Tweet` dataclass (not a raw dict) so every record has a guaranteed
  shape and typos in field names fail immediately instead of silently
  producing missing columns later in the pipeline.
- A `set()` of tweet IDs kept in memory for O(1) duplicate checks while
  scrolling, so we never append the same tweet twice within a run.
"""
from __future__ import annotations

import random
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Set

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from src.scraper import selectors
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A handful of realistic desktop user-agent strings. Rotating between a
# small fixed list is enough to avoid the single most obvious
# fingerprint (every request from the exact same UA string); this is
# NOT an attempt to defeat X's bot-detection, just to not look like a
# default headless-Chrome fingerprint.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]

HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
MENTION_RE = re.compile(r"@(\w+)", re.UNICODE)


@dataclass
class Tweet:
    """One scraped tweet, normalized to a fixed set of fields."""

    tweet_id: str
    username: str
    content: str
    timestamp: Optional[str]      # ISO-8601 string, or None if unparseable
    likes: int
    retweets: int
    replies: int
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    source_hashtag: str = ""      # which search page this came from
    scraped_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_count(text: str) -> int:
    """Turn X's abbreviated counters ('1.2K', '3.4M', '', '12') into an int.

    X shows engagement counts as compact strings. A naive int(text) would
    crash on '1.2K', so this handles the common suffixes explicitly.
    """
    if not text:
        return 0
    text = text.strip().replace(",", "")
    if not text:
        return 0
    multiplier = 1
    if text[-1].upper() == "K":
        multiplier = 1_000
        text = text[:-1]
    elif text[-1].upper() == "M":
        multiplier = 1_000_000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def _normalize_text(raw: str) -> str:
    """Normalize Unicode so Hindi/Marathi/Gujarati text and emoji survive
    the pipeline consistently (NFC form), and collapse stray whitespace
    left behind by X's DOM (line breaks inside a tweet become real
    newlines in the HTML)."""
    text = unicodedata.normalize("NFC", raw)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class TwitterScraper:
    """Scrapes a single hashtag's live search results page.

    One instance = one browser = one hashtag. Running several hashtags
    concurrently means creating several TwitterScraper instances and
    driving them from a thread pool (see main.py) - each Selenium
    WebDriver is single-threaded, so this is the correct level to
    parallelize at, rather than trying to share one browser across
    threads.
    """

    SEARCH_URL = "https://x.com/search?q=%23{hashtag}&src=typed_query&f=live"

    def __init__(self, config: dict):
        self.config = config["scraper"]
        self.driver: Optional[webdriver.Chrome] = None
        self._seen_ids: Set[str] = set()

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        options = Options()
        if self.config.get("headless", True):
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1400,1000")
        options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
        # Reduces the most obvious "navigator.webdriver == true" tell.
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(30)
        logger.info("Chrome driver started (headless=%s)", self.config.get("headless", True))

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.quit()
            except WebDriverException:
                pass
            logger.info("Chrome driver closed")

    # -- core scraping ---------------------------------------------------
    def _random_delay(self) -> None:
        low = self.config.get("min_delay_seconds", 2.5)
        high = self.config.get("max_delay_seconds", 5.5)
        time.sleep(random.uniform(low, high))

    def _hit_login_wall(self) -> bool:
        try:
            self.driver.find_element(By.CSS_SELECTOR, selectors.LOGIN_WALL_MARKER)
            return True
        except NoSuchElementException:
            return False

    def _parse_tweet_element(self, element, hashtag: str) -> Optional[Tweet]:
        try:
            content_el = element.find_element(By.CSS_SELECTOR, selectors.TWEET_TEXT)
            content = _normalize_text(content_el.text)
        except NoSuchElementException:
            # Media-only tweet with no text body - not useful for our
            # text-to-signal pipeline, skip it rather than store an
            # empty row.
            return None

        try:
            username = element.find_element(
                By.CSS_SELECTOR, selectors.TWEET_USERNAME
            ).text.split("\n")[0]
        except NoSuchElementException:
            username = "unknown"

        try:
            timestamp_raw = element.find_element(
                By.CSS_SELECTOR, selectors.TWEET_TIMESTAMP
            ).get_attribute("datetime")
        except NoSuchElementException:
            timestamp_raw = None

        def _count(selector: str) -> int:
            try:
                el = element.find_element(By.CSS_SELECTOR, selector)
                label = el.get_attribute("aria-label") or el.text
                match = re.search(r"[\d.,]+[KM]?", label)
                return _parse_count(match.group()) if match else 0
            except NoSuchElementException:
                return 0

        likes = _count(selectors.TWEET_LIKE_COUNT)
        retweets = _count(selectors.TWEET_RETWEET_COUNT)
        replies = _count(selectors.TWEET_REPLY_COUNT)

        # A tweet has no reliable numeric ID visible in the DOM without
        # opening the permalink, so we build a stable synthetic ID from
        # username + timestamp + a slice of content. Good enough for
        # within-run and cross-run deduplication (see processing/dedup.py
        # for the second, fuzzier layer of dedup on top of this).
        tweet_id = f"{username}|{timestamp_raw}|{content[:40]}"

        return Tweet(
            tweet_id=tweet_id,
            username=username,
            content=content,
            timestamp=timestamp_raw,
            likes=likes,
            retweets=retweets,
            replies=replies,
            hashtags=HASHTAG_RE.findall(content),
            mentions=MENTION_RE.findall(content),
            source_hashtag=hashtag,
        )

    def scrape_hashtag(self, hashtag: str, target_count: int) -> List[Tweet]:
        """Scroll a hashtag's live-search page, collecting tweets until
        `target_count` is reached, the login wall appears, or scrolling
        stops turning up anything new."""
        assert self.driver is not None, "call start() before scrape_hashtag()"
        results: List[Tweet] = []
        url = self.SEARCH_URL.format(hashtag=hashtag)
        logger.info("Opening %s", url)

        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selectors.TWEET_ARTICLE))
            )
        except TimeoutException:
            logger.warning("No tweets loaded for #%s within timeout - page may be rate-limited", hashtag)
            return results

        empty_scrolls = 0
        max_empty = self.config.get("max_empty_scrolls", 8)

        while len(results) < target_count and empty_scrolls < max_empty:
            if self._hit_login_wall():
                logger.info("Login wall reached for #%s after %d tweets", hashtag, len(results))
                break

            articles = self.driver.find_elements(By.CSS_SELECTOR, selectors.TWEET_ARTICLE)
            new_in_pass = 0
            for article in articles:
                try:
                    tweet = self._parse_tweet_element(article, hashtag)
                except Exception as exc:  # noqa: BLE001 - a single bad tweet must not kill the run
                    logger.debug("Skipping unparsable tweet element: %s", exc)
                    continue
                if tweet is None or tweet.tweet_id in self._seen_ids:
                    continue
                self._seen_ids.add(tweet.tweet_id)
                results.append(tweet)
                new_in_pass += 1
                if len(results) >= target_count:
                    break

            empty_scrolls = 0 if new_in_pass > 0 else empty_scrolls + 1

            self.driver.execute_script("window.scrollBy(0, window.innerHeight * 2);")
            self._random_delay()

        logger.info("Collected %d tweets for #%s", len(results), hashtag)
        return results
