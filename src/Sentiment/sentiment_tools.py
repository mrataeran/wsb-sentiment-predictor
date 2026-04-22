"""
Fetch the top N r/wallstreetbets posts and predict a buy / not-buy signal
using FINBert sentiment analysis weighted by upvote scores.
"""

import os

import numpy as np
import praw
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

_FINBERT_MODEL = "ProsusAI/finbert"
_TOP_COMMENTS_PER_POST = 5
_BUY_THRESHOLD = 0.0

_pipeline_cache = None


def _get_finbert_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL)
        device = 0 if torch.cuda.is_available() else -1
        _pipeline_cache = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,
            truncation=True,
            max_length=512,
        )
    return _pipeline_cache


def _make_reddit_client(client_id=None, client_secret=None, user_agent=None):
    return praw.Reddit(
        client_id=client_id or os.environ["WSB_REDDIT_CLIENT_ID"],
        client_secret=client_secret or os.environ["WSB_REDDIT_CLIENT_SECRET"],
        user_agent=user_agent or os.getenv(
            "WSB_REDDIT_USER_AGENT", "wsb-sentiment-predictor/0.1"
        ),
    )


def fetch_wsb_posts(
    client_id=None,
    client_secret=None,
    user_agent=None,
    n_posts=10,
    time_filter="day",
):
    """Return the top *n_posts* from r/wallstreetbets as a list of dicts.

    Each dict contains:
        title, selftext, score, upvote_ratio, num_comments, comments

    Reddit credentials are read from *client_id* / *client_secret* /
    *user_agent* arguments or from the environment variables
    ``WSB_REDDIT_CLIENT_ID``, ``WSB_REDDIT_CLIENT_SECRET``, and
    ``WSB_REDDIT_USER_AGENT``.
    """
    reddit = _make_reddit_client(client_id, client_secret, user_agent)
    subreddit = reddit.subreddit("wallstreetbets")

    posts = []
    for submission in subreddit.top(time_filter=time_filter, limit=n_posts):
        submission.comments.replace_more(limit=0)
        top_comments = sorted(
            [c for c in submission.comments.list() if hasattr(c, "body")],
            key=lambda c: c.score,
            reverse=True,
        )[:_TOP_COMMENTS_PER_POST]

        posts.append(
            {
                "title": submission.title,
                "selftext": submission.selftext,
                "score": max(submission.score, 1),
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "comments": [
                    {"body": c.body, "score": max(c.score, 1)}
                    for c in top_comments
                ],
            }
        )
    return posts


def _net_sentiment(text, pipe):
    """Return a net sentiment score in [-1, 1] for *text* using FINBert.

    Score = P(positive) - P(negative).  Neutral text scores near 0.
    """
    if not text or not text.strip():
        return 0.0
    results = pipe(text)[0]  # list of {label, score}
    by_label = {r["label"].lower(): r["score"] for r in results}
    return by_label.get("positive", 0.0) - by_label.get("negative", 0.0)


def analyze_posts(posts):
    """Run FINBert over a list of post dicts returned by :func:`fetch_wsb_posts`.

    Adds a ``sentiment_score`` key (float in [-1, 1]) to every post dict and
    returns the augmented list.  Score combines the post body (60 %) and its
    top comments weighted by comment upvotes (40 %).
    """
    pipe = _get_finbert_pipeline()
    for post in posts:
        body_text = f"{post['title']} {post['selftext']}".strip()
        body_net = _net_sentiment(body_text, pipe)

        comment_pairs = [
            (_net_sentiment(c["body"], pipe), c["score"])
            for c in post.get("comments", [])
        ]
        if comment_pairs:
            total_w = sum(w for _, w in comment_pairs)
            comment_net = sum(s * w for s, w in comment_pairs) / total_w
        else:
            comment_net = 0.0

        post["sentiment_score"] = 0.6 * body_net + 0.4 * comment_net

    return posts


def predict(
    client_id=None,
    client_secret=None,
    user_agent=None,
    n_posts=10,
    time_filter="day",
    threshold=_BUY_THRESHOLD,
):
    """Fetch the top WSB posts and return a buy / not-buy signal.

    Parameters
    ----------
    client_id, client_secret, user_agent:
        Reddit API credentials (falls back to environment variables).
    n_posts:
        Number of top posts to analyse (default 10).
    time_filter:
        Reddit time filter – one of ``"hour"``, ``"day"``, ``"week"``,
        ``"month"``, ``"year"``, ``"all"`` (default ``"day"``).
    threshold:
        Minimum weighted-sentiment score required for a ``"buy"`` signal
        (default 0.0, i.e. net positive sentiment wins).

    Returns
    -------
    dict
        ``signal``              – ``"buy"`` or ``"not buy"``
        ``weighted_sentiment``  – aggregate sentiment score in [-1, 1]
        ``posts_analyzed``      – number of posts processed
        ``breakdown``           – per-post title, upvote score, and sentiment
    """
    posts = fetch_wsb_posts(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        n_posts=n_posts,
        time_filter=time_filter,
    )
    analyzed = analyze_posts(posts)

    total_score = sum(p["score"] for p in analyzed)
    if total_score == 0:
        weighted_sentiment = float(np.mean([p["sentiment_score"] for p in analyzed]))
    else:
        weighted_sentiment = float(
            sum(p["sentiment_score"] * p["score"] / total_score for p in analyzed)
        )

    return {
        "signal": "buy" if weighted_sentiment > threshold else "not buy",
        "weighted_sentiment": round(weighted_sentiment, 4),
        "posts_analyzed": len(analyzed),
        "breakdown": [
            {
                "title": p["title"],
                "upvote_score": p["score"],
                "sentiment_score": round(p["sentiment_score"], 4),
            }
            for p in analyzed
        ],
    }
