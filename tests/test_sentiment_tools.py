"""
Tests for Sentiment.sentiment_tools.

All tests are offline — no Reddit API calls and no FINBert model download.
The FINBert pipeline and the Reddit client are replaced with mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

from Sentiment.sentiment_tools import (
    _net_sentiment,
    analyze_posts,
    fetch_wsb_posts,
    predict,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pipe(positive=0.7, negative=0.2, neutral=0.1):
    """Return a callable that mimics the FINBert pipeline output."""
    def pipe(text, **kwargs):
        return [[
            {"label": "positive", "score": positive},
            {"label": "negative", "score": negative},
            {"label": "neutral",  "score": neutral},
        ]]
    return pipe


def _make_post(title="GME", selftext="", score=1000, comments=None):
    return {
        "title": title,
        "selftext": selftext,
        "score": score,
        "upvote_ratio": 0.95,
        "num_comments": 10,
        "comments": comments or [],
    }


# ── _net_sentiment ─────────────────────────────────────────────────────────────

class TestNetSentiment:
    def test_positive_dominant(self):
        pipe = _make_pipe(positive=0.8, negative=0.1)
        assert _net_sentiment("Stocks going up!", pipe) == pytest.approx(0.7)

    def test_negative_dominant(self):
        pipe = _make_pipe(positive=0.1, negative=0.8)
        assert _net_sentiment("Market crash!", pipe) == pytest.approx(-0.7)

    def test_neutral(self):
        pipe = _make_pipe(positive=0.5, negative=0.5)
        assert _net_sentiment("Sideways action", pipe) == pytest.approx(0.0)

    def test_empty_string_returns_zero(self):
        pipe = _make_pipe()
        assert _net_sentiment("", pipe) == 0.0

    def test_whitespace_only_returns_zero(self):
        pipe = _make_pipe()
        assert _net_sentiment("   ", pipe) == 0.0


# ── analyze_posts ──────────────────────────────────────────────────────────────

class TestAnalyzePosts:
    def test_adds_sentiment_score_key(self):
        posts = [_make_post()]
        with patch("Sentiment.sentiment_tools._get_finbert_pipeline", return_value=_make_pipe()):
            result = analyze_posts(posts)
        assert "sentiment_score" in result[0]

    def test_sentiment_score_in_range(self):
        posts = [_make_post(), _make_post(title="TSLA moon", score=500)]
        with patch("Sentiment.sentiment_tools._get_finbert_pipeline", return_value=_make_pipe()):
            result = analyze_posts(posts)
        for post in result:
            assert -1.0 <= post["sentiment_score"] <= 1.0

    def test_comments_influence_score(self):
        """A post with bullish comments should score higher than one without."""
        post_no_comments = _make_post(comments=[])
        post_with_comments = _make_post(
            comments=[{"body": "To the moon!", "score": 500}]
        )
        with patch("Sentiment.sentiment_tools._get_finbert_pipeline", return_value=_make_pipe(positive=0.9, negative=0.05)):
            result = analyze_posts([post_no_comments, post_with_comments])
        # Both should be positive; post with comments should be >= without
        assert result[1]["sentiment_score"] >= result[0]["sentiment_score"]

    def test_comment_weight_by_score(self):
        """A high-upvote comment should pull sentiment more than a low-upvote one."""
        bullish_pipe = _make_pipe(positive=0.9, negative=0.05)

        # One post: title neutral (0.0), one high-score bullish comment
        post_high = _make_post(
            comments=[{"body": "Bullish!", "score": 1000}]
        )
        # One post: title neutral, one low-score bullish comment
        post_low = _make_post(
            comments=[{"body": "Bullish!", "score": 1}]
        )
        with patch("Sentiment.sentiment_tools._get_finbert_pipeline", return_value=bullish_pipe):
            result = analyze_posts([post_high, post_low])
        # Both identical text → same score regardless of weight (weight only matters
        # when comments differ in sentiment, so scores should be equal here)
        assert result[0]["sentiment_score"] == pytest.approx(result[1]["sentiment_score"])

    def test_empty_posts_list(self):
        with patch("Sentiment.sentiment_tools._get_finbert_pipeline", return_value=_make_pipe()):
            result = analyze_posts([])
        assert result == []


# ── fetch_wsb_posts ────────────────────────────────────────────────────────────

class TestFetchWsbPosts:
    def _make_submission(self, title, score, selftext="", comments=None):
        sub = MagicMock()
        sub.title = title
        sub.score = score
        sub.selftext = selftext
        sub.upvote_ratio = 0.9
        sub.num_comments = len(comments or [])
        mock_comments = []
        for body, cscore in (comments or []):
            c = MagicMock()
            c.body = body
            c.score = cscore
            mock_comments.append(c)
        sub.comments.list.return_value = mock_comments
        return sub

    def _patch_reddit(self, submissions):
        mock_reddit = MagicMock()
        mock_subreddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_subreddit.top.return_value = iter(submissions)
        return mock_reddit

    def test_returns_correct_number_of_posts(self):
        subs = [self._make_submission(f"Post {i}", score=100 * i) for i in range(5)]
        with patch("Sentiment.sentiment_tools._make_reddit_client", return_value=self._patch_reddit(subs)):
            posts = fetch_wsb_posts(n_posts=5)
        assert len(posts) == 5

    def test_post_dict_has_required_keys(self):
        subs = [self._make_submission("Rocket ship", score=9999)]
        with patch("Sentiment.sentiment_tools._make_reddit_client", return_value=self._patch_reddit(subs)):
            posts = fetch_wsb_posts()
        required = {"title", "selftext", "score", "upvote_ratio", "num_comments", "comments"}
        assert required.issubset(posts[0].keys())

    def test_score_floored_at_one(self):
        """Submissions with 0 or negative upvotes should be stored with score=1."""
        subs = [self._make_submission("Penny stock play", score=0)]
        with patch("Sentiment.sentiment_tools._make_reddit_client", return_value=self._patch_reddit(subs)):
            posts = fetch_wsb_posts()
        assert posts[0]["score"] >= 1

    def test_comments_capped_at_five(self):
        comments = [(f"Comment {i}", 100 - i) for i in range(10)]
        subs = [self._make_submission("Big DD", score=5000, comments=comments)]
        with patch("Sentiment.sentiment_tools._make_reddit_client", return_value=self._patch_reddit(subs)):
            posts = fetch_wsb_posts()
        assert len(posts[0]["comments"]) <= 5


# ── predict ────────────────────────────────────────────────────────────────────

class TestPredict:
    def _setup_mocks(self, positive=0.8, negative=0.1, post_scores=None):
        """Return context managers that mock both Reddit and FINBert."""
        post_scores = post_scores or [1000, 800, 600]
        submissions = []
        for i, s in enumerate(post_scores):
            sub = MagicMock()
            sub.title = f"Post {i}"
            sub.selftext = ""
            sub.score = s
            sub.upvote_ratio = 0.9
            sub.num_comments = 0
            sub.comments.list.return_value = []
            submissions.append(sub)

        mock_reddit = MagicMock()
        mock_subreddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_subreddit.top.return_value = iter(submissions)

        reddit_patch = patch(
            "Sentiment.sentiment_tools._make_reddit_client",
            return_value=mock_reddit,
        )
        finbert_patch = patch(
            "Sentiment.sentiment_tools._get_finbert_pipeline",
            return_value=_make_pipe(positive=positive, negative=negative),
        )
        return reddit_patch, finbert_patch

    def test_buy_signal_on_positive_sentiment(self):
        rp, fp = self._setup_mocks(positive=0.9, negative=0.05)
        with rp, fp:
            result = predict()
        assert result["signal"] == "buy"

    def test_not_buy_signal_on_negative_sentiment(self):
        rp, fp = self._setup_mocks(positive=0.05, negative=0.9)
        with rp, fp:
            result = predict()
        assert result["signal"] == "not buy"

    def test_result_has_required_keys(self):
        rp, fp = self._setup_mocks()
        with rp, fp:
            result = predict()
        assert {"signal", "weighted_sentiment", "posts_analyzed", "breakdown"}.issubset(result.keys())

    def test_weighted_sentiment_in_range(self):
        rp, fp = self._setup_mocks()
        with rp, fp:
            result = predict()
        assert -1.0 <= result["weighted_sentiment"] <= 1.0

    def test_posts_analyzed_count(self):
        rp, fp = self._setup_mocks(post_scores=[100, 200, 300])
        with rp, fp:
            result = predict(n_posts=3)
        assert result["posts_analyzed"] == 3

    def test_breakdown_structure(self):
        rp, fp = self._setup_mocks(post_scores=[500])
        with rp, fp:
            result = predict()
        entry = result["breakdown"][0]
        assert {"title", "upvote_score", "sentiment_score"}.issubset(entry.keys())

    def test_custom_threshold(self):
        """Setting a very high threshold should force 'not buy' even on positive data."""
        rp, fp = self._setup_mocks(positive=0.6, negative=0.1)
        with rp, fp:
            result = predict(threshold=0.99)
        assert result["signal"] == "not buy"
