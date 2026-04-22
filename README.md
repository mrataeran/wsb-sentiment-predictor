# wsb-sentiment-predictor

````md id="q7nd51"
## 1. Install the package

```bash
cd "/Volumes/RayCue/Home/School/STAT 386/wsb-sentiment-predictor"
pip install -e ".[dev]"
````

This installs:

* `praw`
* `transformers`
* `torch`
* `numpy`
* the `Sentiment` package itself in editable mode

---

## 2. Get Reddit API credentials

1. Go to [https://reddit.com/prefs/apps](https://reddit.com/prefs/apps)
2. Click **Create another app** → choose **script**
3. Copy your `client_id` (under the app name) and `client_secret`

Set them as environment variables so you don't hard-code them:

```bash
export WSB_REDDIT_CLIENT_ID="your_id_here"
export WSB_REDDIT_CLIENT_SECRET="your_secret_here"
```

```
```


## sentiment_tools
## Three public functions form the API:

| Function | What it does |
|---|---|
| `fetch_wsb_posts(...)` | Uses PRAW to pull the top **N** posts from `r/wallstreetbets`, including title, body, upvote score, upvote ratio, and the top 5 comments per post (sorted by upvotes). |
| `analyze_posts(posts)` | Runs each post (title + body) and its top comments through FINBERT; combines them as **60% body / 40% comment sentiment weighted by comment upvote scores**. |
| `predict(...)` | Calls both of the above, then aggregates per-post sentiment weighted by each post’s upvote score and returns `"buy"` or `"not buy"`. |

## Sentiment scoring

FINBERT outputs probabilities for `positive`, `negative`, and `neutral`.

The net score is:

`P(positive) - P(negative)`

Range: `[-1, 1]`

A positive aggregate ⇒ `"buy"`.

## FINBERT model

`ProsusAI/finbert` is loaded lazily on first call and cached for the process lifetime.

It respects the 512-token limit via:

`truncation=True`

## Reddit credentials

Credentials can be passed directly or set as environment variables:

- `WSB_REDDIT_CLIENT_ID`
- `WSB_REDDIT_CLIENT_SECRET`

## pyproject.toml

Wired up with `hatchling` so the package is installable with:

- `pip install -e .`
- `pip install .`

## Declared dependencies

- `praw`
- `transformers`
- `torch`
- `numpy`