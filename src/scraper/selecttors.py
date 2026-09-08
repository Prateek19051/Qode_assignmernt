"""
CSS selectors used to find elements on the X (Twitter) search results page.

Why this file exists on its own: X changes its front-end HTML fairly
often. Keeping every selector string in one place means that when a
selector breaks, there is exactly one file to update instead of hunting
through the scraping logic to find where the string is buried.

These selectors target the public search page:
https://x.com/search?q=<hashtag>&src=typed_query&f=live
No login is required to view public search results, but X may still
show a login wall after a number of tweets are viewed - see the
"login wall" handling in twitter_scraper.py.
"""

TWEET_ARTICLE = "article[data-testid='tweet']"
TWEET_TEXT = "div[data-testid='tweetText']"
TWEET_USERNAME = "div[data-testid='User-Name'] a[role='link']"
TWEET_TIMESTAMP = "time"
TWEET_REPLY_COUNT = "div[data-testid='reply']"
TWEET_RETWEET_COUNT = "div[data-testid='retweet']"
TWEET_LIKE_COUNT = "div[data-testid='like']"

LOGIN_WALL_MARKER = "div[data-testid='loginButton']"
