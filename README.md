# wsb-sentiment-predictor

A Python package that fetches the top posts from **r/wallstreetbets**, runs financial sentiment analysis using **FINBert**, and outputs a **"buy"** or **"not buy"** signal for **S&P 100 stocks**.

You can get a signal for a single stock, scan all S&P 100 tickers mentioned across the top WSB posts at once, or run a general overall-market sentiment check.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Installation](#installation)
3. [Reddit API Setup](#reddit-api-setup)
4. [Quick Start](#quick-start)
5. [Step-by-Step Tutorial](#step-by-step-tutorial)
6. [Function Reference](#function-reference)
7. [Running the Tests](#running-the-tests)

---

## How It Works

The package offers two main modes:

**Mode 1 — Single ticker** (`predict_ticker`)
```
Search WSB for posts mentioning "NVDA"
        │
        ▼
  Post title + body ──┐
  Top 5 comments      ├──► FINBert ──► net sentiment per post
  (weighted by votes) ┘               (P(positive) − P(negative))
        │
        ▼
  Aggregate across posts weighted by upvote score
        │
        ▼
  weighted_sentiment > 0  →  "buy"
  weighted_sentiment ≤ 0  →  "not buy"
```

**Mode 2 — All S&P 100 tickers** (`predict_sp100`)
```
Fetch top N WSB posts
        │
        ▼
  Scan each post for S&P 100 ticker mentions ($AAPL, MSFT, …)
        │
        ▼
  Run FINBert once per post, then bucket results by ticker
        │
        ▼
  { "AAPL": "buy", "MSFT": "not buy", "NVDA": "buy", … }
```

**FINBert** (`ProsusAI/finbert`) is a BERT model fine-tuned on financial news. It outputs three class probabilities — `positive`, `negative`, and `neutral` — from which a net score in **[−1, 1]** is computed as `P(positive) − P(negative)`.

Each post's sentiment combines **60% post body** and **40% comment sentiment** (comments weighted by their own upvote scores). Posts with more upvotes carry more weight in the final aggregate.

---

## Installation

### Prerequisites

- Python 3.9 or later
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Install with uv (recommended)

```bash
git clone https://github.com/<your-username>/wsb-sentiment-predictor.git
cd wsb-sentiment-predictor
uv venv
uv pip install -e .
```

### Install with pip

```bash
git clone https://github.com/<your-username>/wsb-sentiment-predictor.git
cd wsb-sentiment-predictor
pip install -e .
```

This installs the `Sentiment` package and all required dependencies:

| Dependency | Purpose |
|---|---|
| `praw` | Reddit API client |
| `transformers` | FINBert model loading and inference |
| `torch` | PyTorch backend for FINBert |
| `numpy` | Weighted aggregation |

> **Note:** The first time you run sentiment analysis, `transformers` will download the FINBert model weights (~400 MB). Subsequent calls in the same process use the cached model.

---

## Reddit API Setup

The package uses the [Reddit API](https://www.reddit.com/dev/api/) via PRAW. You need a free Reddit developer account to get credentials.

**Step 1 — Create a Reddit app**

1. Log in to Reddit and go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Scroll to the bottom and click **"create another app"**
3. Fill in any name (e.g., `wsb-predictor`), select **script**, and click **create app**

**Step 2 — Copy your credentials**

- `client_id` — the short string directly under your app name
- `client_secret` — the longer string labeled "secret"

Learn how to get credentials at [geeksforgeeks](https://www.geeksforgeeks.org/python/how-to-get-client_id-and-client_secret-for-python-reddit-api-registration/).

**Step 3 — Set environment variables**

```bash
export WSB_REDDIT_CLIENT_ID="your_client_id"
export WSB_REDDIT_CLIENT_SECRET="your_client_secret"
```

To make these permanent, add the two lines above to your `~/.zshrc` or `~/.bashrc`.

Alternatively, pass credentials directly as arguments to any function (see [Function Reference](#function-reference)).

---

## Quick Start

```python
from Sentiment import predict_ticker, predict_sp100

# Signal for one S&P 100 stock
result = predict_ticker("NVDA")
print(result["signal"])              # "buy" or "not buy"
print(result["weighted_sentiment"])  # e.g. 0.312

# Signals for every S&P 100 stock mentioned in today's top WSB posts
signals = predict_sp100()
for ticker, info in signals.items():
    print(f"{ticker}: {info['signal']}  ({info['weighted_sentiment']:+.3f})")
```

---

## Step-by-Step Tutorial

This section walks through each layer of the package so you can inspect and understand what's happening at every step.

### Step 1 — Fetch posts from Reddit

```python
from Sentiment import fetch_wsb_posts

posts = fetch_wsb_posts(n_posts=5, time_filter="day")

post = posts[0]
print("Title:    ", post["title"])
print("Upvotes:  ", post["score"])
print("Comments: ", post["num_comments"])
print("Top comment:", post["comments"][0]["body"][:80])
```

Each element in `posts` is a dictionary:

```python
{
    "title":        "NVDA earnings beat — what now?",
    "selftext":     "...",          # post body (empty for link posts)
    "score":        14200,          # upvote count
    "upvote_ratio": 0.94,
    "num_comments": 3041,
    "comments": [                   # top 5 comments by upvote score
        {"body": "Holding through earnings, let's go", "score": 872},
        ...
    ]
}
```

### Step 2 — Run FINBert sentiment analysis

```python
from Sentiment import fetch_wsb_posts, analyze_posts

posts = fetch_wsb_posts(n_posts=5)
analyzed = analyze_posts(posts)     # downloads FINBert on first call

for post in analyzed:
    print(f"{post['score']:>6} upvotes | {post['sentiment_score']:+.3f} | {post['title'][:50]}")
```

Example output:

```
 14200 upvotes | +0.412 | NVDA earnings beat — what now?
  8900 upvotes | -0.231 | My portfolio is down 60%
  7100 upvotes | +0.088 | Daily Discussion Thread
```

`sentiment_score` is a float in **[−1, 1]**:
- `+1.0` — strongly positive (bullish)
- ` 0.0` — neutral
- `−1.0` — strongly negative (bearish)

### Step 3 — Get a signal for one S&P 100 stock

Use `predict_ticker` to search WSB specifically for posts discussing a stock and get its buy / not-buy signal.

```python
from Sentiment import predict_ticker

result = predict_ticker("NVDA", n_posts=10, time_filter="week")

print("Ticker:            ", result["ticker"])
print("Signal:            ", result["signal"])
print("Weighted sentiment:", result["weighted_sentiment"])
print()
print("Per-post breakdown:")
for entry in result["breakdown"]:
    print(f"  [{entry['upvote_score']:>6} pts | {entry['sentiment_score']:+.3f}]  {entry['title'][:55]}")
```

Example output:

```
Ticker:             NVDA
Signal:             buy
Weighted sentiment: 0.3124

Per-post breakdown:
  [ 14200 pts | +0.412]  NVDA earnings beat — what now?
  [  9300 pts | +0.289]  Why I'm still holding $NVDA into next quarter
  [  4100 pts | -0.071]  NVDA puts printing, don't say I didn't warn you
  ...
```

### Step 4 — Scan all S&P 100 tickers at once

Use `predict_sp100` to fetch a broad set of top WSB posts, automatically detect every S&P 100 ticker mentioned, and return a signal for each one.

```python
from Sentiment import predict_sp100

signals = predict_sp100(n_posts=50, time_filter="day")

for ticker, info in sorted(signals.items()):
    print(f"{ticker:<6} {info['signal']:<10} sentiment={info['weighted_sentiment']:+.3f}  posts={info['posts_analyzed']}")
```

Example output:

```
AAPL   buy        sentiment=+0.183  posts=3
AMZN   not buy    sentiment=-0.042  posts=1
NVDA   buy        sentiment=+0.312  posts=7
TSLA   not buy    sentiment=-0.201  posts=4
...
```

Only tickers that appear in at least one of the fetched posts will be included in the result.

### Step 5 — Check which tickers are in the S&P 100 list

```python
from Sentiment import SP100_TICKERS

print("NVDA" in SP100_TICKERS)   # True
print("GME" in SP100_TICKERS)    # False — not in the S&P 100
print(len(SP100_TICKERS))        # 96
```

### Step 6 — Customize the prediction

```python
# Analyse the past week instead of today
result = predict_ticker("AAPL", time_filter="week")

# Require stronger positive sentiment before signalling "buy"
result = predict_ticker("MSFT", threshold=0.15)

# Scan more posts to catch more tickers
signals = predict_sp100(n_posts=100, time_filter="week")

# Pass credentials explicitly instead of using env vars
result = predict_ticker(
    "JPM",
    client_id="abc123",
    client_secret="xyz789",
)
```

---

## Function Reference

### `SP100_TICKERS`

A `set` of 96 current S&P 100 ticker symbols (e.g. `"AAPL"`, `"NVDA"`, `"MSFT"`). Use this to check whether a ticker is covered before calling `predict_ticker`.

```python
from Sentiment import SP100_TICKERS

"NVDA" in SP100_TICKERS   # True
"GME"  in SP100_TICKERS   # False
```

---

### `fetch_wsb_posts`

Fetches the top posts from r/wallstreetbets.

```python
fetch_wsb_posts(
    client_id=None,
    client_secret=None,
    user_agent=None,
    n_posts=10,
    time_filter="day",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client_id` | `str \| None` | `None` | Reddit app client ID. Falls back to `WSB_REDDIT_CLIENT_ID` env var. |
| `client_secret` | `str \| None` | `None` | Reddit app client secret. Falls back to `WSB_REDDIT_CLIENT_SECRET` env var. |
| `user_agent` | `str \| None` | `None` | Reddit API user agent. Falls back to `WSB_REDDIT_USER_AGENT` env var or a sensible default. |
| `n_posts` | `int` | `10` | Number of top posts to fetch. |
| `time_filter` | `str` | `"day"` | Reddit time window: `"hour"`, `"day"`, `"week"`, `"month"`, `"year"`, or `"all"`. |

**Returns:** `list[dict]` — Each dict contains `title`, `selftext`, `score`, `upvote_ratio`, `num_comments`, and `comments` (list of `{body, score}` dicts).

---

### `analyze_posts`

Runs FINBert sentiment analysis over a list of post dicts.

```python
analyze_posts(posts)
```

| Parameter | Type | Description |
|---|---|---|
| `posts` | `list[dict]` | Output of `fetch_wsb_posts`. |

**Returns:** The same list with a `sentiment_score` key added to every dict. Score is a `float` in `[-1, 1]`: positive means bullish, negative means bearish.

---

### `predict_ticker`

Searches r/wallstreetbets for a specific S&P 100 stock and returns a buy / not-buy signal.

```python
predict_ticker(
    ticker,
    client_id=None,
    client_secret=None,
    user_agent=None,
    n_posts=10,
    time_filter="week",
    threshold=0.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ticker` | `str` | — | An S&P 100 ticker symbol (e.g. `"NVDA"`). Raises `ValueError` if not in `SP100_TICKERS`. |
| `client_id` | `str \| None` | `None` | Reddit app client ID. |
| `client_secret` | `str \| None` | `None` | Reddit app client secret. |
| `user_agent` | `str \| None` | `None` | Reddit API user agent. |
| `n_posts` | `int` | `10` | Number of WSB search results to analyse. |
| `time_filter` | `str` | `"week"` | Reddit time window: `"hour"`, `"day"`, `"week"`, `"month"`, `"year"`, or `"all"`. |
| `threshold` | `float` | `0.0` | Minimum weighted sentiment required for a `"buy"` signal. |

**Returns:** `dict` with the following keys:

| Key | Type | Description |
|---|---|---|
| `ticker` | `str` | The ticker symbol in upper case |
| `signal` | `str` | `"buy"` or `"not buy"` |
| `weighted_sentiment` | `float` | Aggregate sentiment score in `[-1, 1]` |
| `posts_analyzed` | `int` | Number of posts processed |
| `breakdown` | `list[dict]` | Per-post `title`, `upvote_score`, and `sentiment_score` |

---

### `predict_sp100`

Scans the top WSB posts, detects S&P 100 ticker mentions, and returns a per-ticker signal for every mentioned stock.

```python
predict_sp100(
    client_id=None,
    client_secret=None,
    user_agent=None,
    n_posts=50,
    time_filter="day",
    threshold=0.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client_id` | `str \| None` | `None` | Reddit app client ID. |
| `client_secret` | `str \| None` | `None` | Reddit app client secret. |
| `user_agent` | `str \| None` | `None` | Reddit API user agent. |
| `n_posts` | `int` | `50` | Number of top WSB posts to scan for ticker mentions. |
| `time_filter` | `str` | `"day"` | Reddit time window: `"hour"`, `"day"`, `"week"`, `"month"`, `"year"`, or `"all"`. |
| `threshold` | `float` | `0.0` | Minimum weighted sentiment required for a `"buy"` signal. |

**Returns:** `dict[str, dict]` — Mapping of ticker symbol → signal dict. Only tickers that appear in at least one fetched post are included. Each value contains:

| Key | Type | Description |
|---|---|---|
| `signal` | `str` | `"buy"` or `"not buy"` |
| `weighted_sentiment` | `float` | Aggregate sentiment score in `[-1, 1]` |
| `posts_analyzed` | `int` | Number of posts that mentioned this ticker |
| `breakdown` | `list[dict]` | Per-post `title`, `upvote_score`, and `sentiment_score` |

---

### `predict`

Returns a general WSB market-sentiment signal without filtering by any specific ticker. Useful as a broad market mood indicator.

```python
predict(
    client_id=None,
    client_secret=None,
    user_agent=None,
    n_posts=10,
    time_filter="day",
    threshold=0.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client_id` | `str \| None` | `None` | Reddit app client ID. |
| `client_secret` | `str \| None` | `None` | Reddit app client secret. |
| `user_agent` | `str \| None` | `None` | Reddit API user agent. |
| `n_posts` | `int` | `10` | Number of top posts to analyse. |
| `time_filter` | `str` | `"day"` | Reddit time window (same options as `fetch_wsb_posts`). |
| `threshold` | `float` | `0.0` | Minimum weighted sentiment required for a `"buy"` signal. |

**Returns:** `dict` with keys `signal`, `weighted_sentiment`, `posts_analyzed`, and `breakdown` (same shape as `predict_ticker` minus the `ticker` key).

---

## Running the Tests

The test suite is fully offline — no Reddit credentials or model download needed.

```bash
uv run pytest tests/ -v
```

Expected output:

```
collected 21 items

tests/test_sentiment_tools.py::TestNetSentiment::test_positive_dominant PASSED
tests/test_sentiment_tools.py::TestNetSentiment::test_negative_dominant PASSED
...
============================== 21 passed in 1.96s ==============================
```
