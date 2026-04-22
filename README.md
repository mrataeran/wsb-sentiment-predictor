# wsb-sentiment-predictor

A Python package that fetches the top posts from **r/wallstreetbets**, runs financial sentiment analysis using **FINBert**, and outputs a simple **"buy"** or **"not buy"** signal.

Sentiment is weighted by each post's upvote score and its top comments, giving more influential posts a stronger voice in the final prediction.

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

```
r/wallstreetbets (top N posts)
        │
        ▼
  Post title + body ──┐
  Top 5 comments      ├──► FINBert ──► net sentiment score per post
  (weighted by votes) ┘               (P(positive) − P(negative))
        │
        ▼
  Aggregate across posts
  (weighted by upvote score)
        │
        ▼
  weighted_sentiment > 0  →  "buy"
  weighted_sentiment ≤ 0  →  "not buy"
```

**FINBert** (`ProsusAI/finbert`) is a BERT model fine-tuned on financial news. It outputs three class probabilities — `positive`, `negative`, and `neutral` — from which a net score in **[−1, 1]** is computed.

Each post's sentiment combines **60% post body** and **40% comment sentiment** (comments themselves are weighted by their own upvote scores). Posts with more upvotes carry more weight in the final aggregate.

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
from Sentiment import predict

result = predict()

print(result["signal"])              # "buy" or "not buy"
print(result["weighted_sentiment"])  # e.g. 0.1823
```

That's it. The function fetches today's top 10 WSB posts, runs FINBert on each one, and returns a signal.

---

## Step-by-Step Tutorial

This section walks through each layer of the package individually so you can inspect and understand what's happening at every step.

### Step 1 — Fetch posts from Reddit

```python
from Sentiment import fetch_wsb_posts

posts = fetch_wsb_posts(n_posts=5, time_filter="day")

# Inspect the first post
post = posts[0]
print("Title:    ", post["title"])
print("Upvotes:  ", post["score"])
print("Comments: ", post["num_comments"])
print("Top comment:", post["comments"][0]["body"][:80])
```

Each element in `posts` is a dictionary:

```python
{
    "title":        "GME short squeeze incoming",
    "selftext":     "...",          # post body (empty for link posts)
    "score":        14200,          # upvote count
    "upvote_ratio": 0.94,
    "num_comments": 3041,
    "comments": [                   # top 5 comments by upvote score
        {"body": "We're all gonna make it", "score": 872},
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
 14200 upvotes | +0.412 | GME short squeeze incoming
  8900 upvotes | -0.231 | My portfolio is down 60%
  7100 upvotes | +0.088 | Daily Discussion Thread
```

`sentiment_score` is a float in **[−1, 1]**:
- `+1.0` → strongly positive (bullish)
- ` 0.0` → neutral
- `−1.0` → strongly negative (bearish)

### Step 3 — Get the final signal

```python
from Sentiment import predict

result = predict(n_posts=10, time_filter="day")

print("Signal:            ", result["signal"])
print("Weighted sentiment:", result["weighted_sentiment"])
print("Posts analyzed:    ", result["posts_analyzed"])
print()
print("Per-post breakdown:")
for entry in result["breakdown"]:
    print(f"  [{entry['upvote_score']:>6} pts | {entry['sentiment_score']:+.3f}]  {entry['title'][:55]}")
```

Example output:

```
Signal:             buy
Weighted sentiment: 0.1714
Posts analyzed:     10

Per-post breakdown:
  [ 14200 pts | +0.412]  GME short squeeze incoming — here's the DD
  [  8900 pts | -0.231]  My portfolio is down 60% AMA
  [  7100 pts | +0.088]  Daily Discussion Thread - April 22, 2026
  ...
```

### Step 4 — Customize the prediction

```python
# Analyse the past week instead of today
result = predict(n_posts=25, time_filter="week")

# Require stronger positive sentiment before signalling "buy"
result = predict(threshold=0.15)

# Pass credentials explicitly instead of using env vars
result = predict(
    client_id="abc123",
    client_secret="xyz789",
    n_posts=10,
)
```

---

## Function Reference

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

### `predict`

End-to-end function: fetches posts, runs sentiment analysis, and returns a signal.

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

**Returns:** `dict` with the following keys:

| Key | Type | Description |
|---|---|---|
| `signal` | `str` | `"buy"` or `"not buy"` |
| `weighted_sentiment` | `float` | Aggregate sentiment score in `[-1, 1]` |
| `posts_analyzed` | `int` | Number of posts that were processed |
| `breakdown` | `list[dict]` | Per-post `title`, `upvote_score`, and `sentiment_score` |

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
