# wsb-sentiment-predictor

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