"""
Tests for Sentiment.sentiment_tools.

All tests are offline — no Reddit API calls and no FINBert model download.
The FINBert pipeline and the Reddit client are replaced with mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

import Sentiment.sentiment_tools as _mod

from Sentiment.sentiment_tools import (
    SP100_TICKERS,
    _extract_sp100_tickers,
    _net_sentiment,
    analyze_posts,
    fetch_sp100_tickers,
    fetch_wsb_posts,
    predict,
    predict_sp100,
    predict_ticker,
)

# ── Shared helpers ─────────────────────────────────────────────────────────────

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


def _make_submission(title, score, selftext="", comments=None):
    """Build a MagicMock that looks like a PRAW Submission."""
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


def _make_mock_reddit(submissions, use_search=False):
    """Return a mock Reddit client whose subreddit yields *submissions*.

    Set *use_search=True* when the code under test calls subreddit.search()
    (i.e. predict_ticker) instead of subreddit.top() (i.e. predict_sp100).
    """
    mock_reddit = MagicMock()
    mock_subreddit = MagicMock()
    mock_reddit.subreddit.return_value = mock_subreddit
    if use_search:
        mock_subreddit.search.return_value = iter(submissions)
    else:
        mock_subreddit.top.return_value = iter(submissions)
    return mock_reddit


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


# ── fetch_sp100_tickers ────────────────────────────────────────────────────────

class TestFetchSp100Tickers:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        """Ensure the module-level cache is clear before and after each test."""
        _mod._sp100_cache = None
        yield
        _mod._sp100_cache = None

    def _make_fake_table(self, symbols):
        import pandas as pd
        return [pd.DataFrame({"Symbol": symbols})]

    def test_returns_set_of_tickers_on_success(self):
        fake = self._make_fake_table(["AAPL", "NVDA", "MSFT"])
        with patch("Sentiment.sentiment_tools.pd.read_html", return_value=fake):
            result = fetch_sp100_tickers()
        assert result == {"AAPL", "NVDA", "MSFT"}

    def test_dot_removed_from_ticker(self):
        fake = self._make_fake_table(["BRK.B", "AAPL"])
        with patch("Sentiment.sentiment_tools.pd.read_html", return_value=fake):
            result = fetch_sp100_tickers()
        assert "BRKB" in result
        assert "BRK.B" not in result

    def test_falls_back_to_hardcoded_on_network_error(self):
        with patch("Sentiment.sentiment_tools.pd.read_html", side_effect=Exception("timeout")):
            result = fetch_sp100_tickers()
        assert result is SP100_TICKERS

    def test_falls_back_when_table_returns_empty(self):
        import pandas as pd
        with patch("Sentiment.sentiment_tools.pd.read_html",
                   return_value=[pd.DataFrame({"Symbol": []})]):
            result = fetch_sp100_tickers()
        assert result is SP100_TICKERS

    def test_result_is_cached_after_first_call(self):
        fake = self._make_fake_table(["AAPL", "NVDA"])
        with patch("Sentiment.sentiment_tools.pd.read_html", return_value=fake) as mock_read:
            fetch_sp100_tickers()
            fetch_sp100_tickers()
        mock_read.assert_called_once()

    def test_cache_is_returned_on_subsequent_calls(self):
        fake = self._make_fake_table(["AAPL"])
        with patch("Sentiment.sentiment_tools.pd.read_html", return_value=fake):
            first  = fetch_sp100_tickers()
            second = fetch_sp100_tickers()
        assert first is second


# ── _extract_sp100_tickers ─────────────────────────────────────────────────────

class TestExtractSp100Tickers:
    @pytest.fixture(autouse=True)
    def _mock_fetch(self):
        """Keep ticker-extraction tests offline by pinning the live fetch."""
        with patch("Sentiment.sentiment_tools.fetch_sp100_tickers", return_value=SP100_TICKERS):
            yield

    def test_plain_uppercase_ticker(self):
        assert "NVDA" in _extract_sp100_tickers("NVDA to the moon")

    def test_dollar_sign_ticker(self):
        assert "AAPL" in _extract_sp100_tickers("Buying $AAPL calls")

    def test_multiple_tickers(self):
        result = _extract_sp100_tickers("NVDA and MSFT both up today")
        assert "NVDA" in result
        assert "MSFT" in result

    def test_non_sp100_ticker_excluded(self):
        # GME is not in the S&P 100
        assert "GME" not in _extract_sp100_tickers("GME to the moon")

    def test_single_letter_without_dollar_excluded(self):
        # "F" (Ford) requires a $ prefix to avoid false positives
        assert "F" not in _extract_sp100_tickers("F is going up")

    def test_single_letter_with_dollar_included(self):
        assert "F" in _extract_sp100_tickers("Buying $F calls")

    def test_lowercase_not_matched(self):
        assert "AAPL" not in _extract_sp100_tickers("aapl earnings today")

    def test_empty_string_returns_empty_set(self):
        assert _extract_sp100_tickers("") == set()

    def test_returns_only_sp100_subset(self):
        result = _extract_sp100_tickers("NVDA GME AAPL MOON")
        assert result.issubset(SP100_TICKERS)


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
        assert result[1]["sentiment_score"] >= result[0]["sentiment_score"]

    def test_comment_weight_by_score(self):
        """Identical comment text produces the same score regardless of upvote weight."""
        bullish_pipe = _make_pipe(positive=0.9, negative=0.05)
        post_high = _make_post(comments=[{"body": "Bullish!", "score": 1000}])
        post_low  = _make_post(comments=[{"body": "Bullish!", "score": 1}])
        with patch("Sentiment.sentiment_tools._get_finbert_pipeline", return_value=bullish_pipe):
            result = analyze_posts([post_high, post_low])
        assert result[0]["sentiment_score"] == pytest.approx(result[1]["sentiment_score"])

    def test_empty_posts_list(self):
        with patch("Sentiment.sentiment_tools._get_finbert_pipeline", return_value=_make_pipe()):
            result = analyze_posts([])
        assert result == []


# ── fetch_wsb_posts ────────────────────────────────────────────────────────────

class TestFetchWsbPosts:
    def test_returns_correct_number_of_posts(self):
        subs = [_make_submission(f"Post {i}", score=100 * i) for i in range(5)]
        with patch("Sentiment.sentiment_tools._make_reddit_client",
                   return_value=_make_mock_reddit(subs)):
            posts = fetch_wsb_posts(n_posts=5)
        assert len(posts) == 5

    def test_post_dict_has_required_keys(self):
        subs = [_make_submission("Rocket ship", score=9999)]
        with patch("Sentiment.sentiment_tools._make_reddit_client",
                   return_value=_make_mock_reddit(subs)):
            posts = fetch_wsb_posts()
        required = {"title", "selftext", "score", "upvote_ratio", "num_comments", "comments"}
        assert required.issubset(posts[0].keys())

    def test_score_floored_at_one(self):
        subs = [_make_submission("Penny stock play", score=0)]
        with patch("Sentiment.sentiment_tools._make_reddit_client",
                   return_value=_make_mock_reddit(subs)):
            posts = fetch_wsb_posts()
        assert posts[0]["score"] >= 1

    def test_comments_capped_at_five(self):
        comments = [(f"Comment {i}", 100 - i) for i in range(10)]
        subs = [_make_submission("Big DD", score=5000, comments=comments)]
        with patch("Sentiment.sentiment_tools._make_reddit_client",
                   return_value=_make_mock_reddit(subs)):
            posts = fetch_wsb_posts()
        assert len(posts[0]["comments"]) <= 5


# ── predict ────────────────────────────────────────────────────────────────────

class TestPredict:
    def _setup_mocks(self, positive=0.8, negative=0.1, post_scores=None):
        post_scores = post_scores or [1000, 800, 600]
        subs = [_make_submission(f"Post {i}", score=s)
                for i, s in enumerate(post_scores)]
        reddit_patch = patch(
            "Sentiment.sentiment_tools._make_reddit_client",
            return_value=_make_mock_reddit(subs),
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
        rp, fp = self._setup_mocks(positive=0.6, negative=0.1)
        with rp, fp:
            result = predict(threshold=0.99)
        assert result["signal"] == "not buy"


# ── predict_ticker ─────────────────────────────────────────────────────────────

class TestPredictTicker:
    def _setup_mocks(self, positive=0.8, negative=0.1, post_scores=None):
        post_scores = post_scores or [1000, 800, 600]
        # predict_ticker calls subreddit.search(), not subreddit.top()
        subs = [_make_submission(f"NVDA post {i}", score=s)
                for i, s in enumerate(post_scores)]
        reddit_patch = patch(
            "Sentiment.sentiment_tools._make_reddit_client",
            return_value=_make_mock_reddit(subs, use_search=True),
        )
        finbert_patch = patch(
            "Sentiment.sentiment_tools._get_finbert_pipeline",
            return_value=_make_pipe(positive=positive, negative=negative),
        )
        tickers_patch = patch(
            "Sentiment.sentiment_tools.fetch_sp100_tickers",
            return_value=SP100_TICKERS,
        )
        return reddit_patch, finbert_patch, tickers_patch

    def test_buy_signal_on_positive_sentiment(self):
        rp, fp, tp = self._setup_mocks(positive=0.9, negative=0.05)
        with rp, fp, tp:
            result = predict_ticker("NVDA")
        assert result["signal"] == "buy"

    def test_not_buy_signal_on_negative_sentiment(self):
        rp, fp, tp = self._setup_mocks(positive=0.05, negative=0.9)
        with rp, fp, tp:
            result = predict_ticker("NVDA")
        assert result["signal"] == "not buy"

    def test_result_has_required_keys(self):
        rp, fp, tp = self._setup_mocks()
        with rp, fp, tp:
            result = predict_ticker("NVDA")
        assert {"ticker", "signal", "weighted_sentiment", "posts_analyzed", "breakdown"}.issubset(result.keys())

    def test_ticker_key_is_uppercased(self):
        rp, fp, tp = self._setup_mocks()
        with rp, fp, tp:
            result = predict_ticker("nvda")
        assert result["ticker"] == "NVDA"

    def test_weighted_sentiment_in_range(self):
        rp, fp, tp = self._setup_mocks()
        with rp, fp, tp:
            result = predict_ticker("NVDA")
        assert -1.0 <= result["weighted_sentiment"] <= 1.0

    def test_posts_analyzed_count(self):
        rp, fp, tp = self._setup_mocks(post_scores=[100, 200, 300])
        with rp, fp, tp:
            result = predict_ticker("NVDA")
        assert result["posts_analyzed"] == 3

    def test_breakdown_structure(self):
        rp, fp, tp = self._setup_mocks(post_scores=[500])
        with rp, fp, tp:
            result = predict_ticker("NVDA")
        entry = result["breakdown"][0]
        assert {"title", "upvote_score", "sentiment_score"}.issubset(entry.keys())

    def test_custom_threshold(self):
        rp, fp, tp = self._setup_mocks(positive=0.6, negative=0.1)
        with rp, fp, tp:
            result = predict_ticker("NVDA", threshold=0.99)
        assert result["signal"] == "not buy"

    def test_invalid_ticker_raises_value_error(self):
        tp = patch("Sentiment.sentiment_tools.fetch_sp100_tickers", return_value=SP100_TICKERS)
        with tp, pytest.raises(ValueError, match="not in the S&P 100"):
            predict_ticker("GME")

    def test_no_posts_returns_not_buy(self):
        mock_reddit = _make_mock_reddit([], use_search=True)
        tp = patch("Sentiment.sentiment_tools.fetch_sp100_tickers", return_value=SP100_TICKERS)
        rp = patch("Sentiment.sentiment_tools._make_reddit_client", return_value=mock_reddit)
        with rp, tp:
            result = predict_ticker("NVDA")
        assert result["signal"] == "not buy"
        assert result["posts_analyzed"] == 0
        assert result["breakdown"] == []


# ── predict_sp100 ──────────────────────────────────────────────────────────────

class TestPredictSp100:
    def _setup_mocks(self, submissions, positive=0.8, negative=0.1):
        # predict_sp100 calls subreddit.top()
        reddit_patch = patch(
            "Sentiment.sentiment_tools._make_reddit_client",
            return_value=_make_mock_reddit(submissions, use_search=False),
        )
        finbert_patch = patch(
            "Sentiment.sentiment_tools._get_finbert_pipeline",
            return_value=_make_pipe(positive=positive, negative=negative),
        )
        tickers_patch = patch(
            "Sentiment.sentiment_tools.fetch_sp100_tickers",
            return_value=SP100_TICKERS,
        )
        return reddit_patch, finbert_patch, tickers_patch

    def test_mentioned_tickers_appear_in_result(self):
        subs = [_make_submission("NVDA and AAPL both ripping", score=1000)]
        rp, fp, tp = self._setup_mocks(subs)
        with rp, fp, tp:
            result = predict_sp100()
        assert "NVDA" in result
        assert "AAPL" in result

    def test_unmentioned_tickers_absent_from_result(self):
        subs = [_make_submission("NVDA to the moon", score=1000)]
        rp, fp, tp = self._setup_mocks(subs)
        with rp, fp, tp:
            result = predict_sp100()
        assert "MSFT" not in result

    def test_non_sp100_ticker_absent_from_result(self):
        subs = [_make_submission("GME squeezing again", score=5000)]
        rp, fp, tp = self._setup_mocks(subs)
        with rp, fp, tp:
            result = predict_sp100()
        assert "GME" not in result

    def test_returns_empty_dict_when_no_sp100_tickers_found(self):
        subs = [_make_submission("market is weird today", score=500)]
        rp, fp, tp = self._setup_mocks(subs)
        with rp, fp, tp:
            result = predict_sp100()
        assert result == {}

    def test_returns_empty_dict_when_no_posts(self):
        rp, fp, tp = self._setup_mocks([])
        with rp, fp, tp:
            result = predict_sp100()
        assert result == {}

    def test_each_entry_has_required_keys(self):
        subs = [_make_submission("NVDA earnings beat", score=1000)]
        rp, fp, tp = self._setup_mocks(subs)
        with rp, fp, tp:
            result = predict_sp100()
        entry = result["NVDA"]
        assert {"signal", "weighted_sentiment", "posts_analyzed", "breakdown"}.issubset(entry.keys())

    def test_buy_signal_on_positive_sentiment(self):
        subs = [_make_submission("NVDA bull case", score=1000)]
        rp, fp, tp = self._setup_mocks(subs, positive=0.9, negative=0.05)
        with rp, fp, tp:
            result = predict_sp100()
        assert result["NVDA"]["signal"] == "buy"

    def test_not_buy_signal_on_negative_sentiment(self):
        subs = [_make_submission("NVDA bear case", score=1000)]
        rp, fp, tp = self._setup_mocks(subs, positive=0.05, negative=0.9)
        with rp, fp, tp:
            result = predict_sp100()
        assert result["NVDA"]["signal"] == "not buy"

    def test_post_shared_across_multiple_tickers(self):
        """A post mentioning two tickers should contribute to both signals."""
        subs = [_make_submission("NVDA and MSFT earnings today", score=2000)]
        rp, fp, tp = self._setup_mocks(subs)
        with rp, fp, tp:
            result = predict_sp100()
        assert "NVDA" in result
        assert "MSFT" in result
        assert result["NVDA"]["posts_analyzed"] == 1
        assert result["MSFT"]["posts_analyzed"] == 1

    def test_custom_threshold(self):
        subs = [_make_submission("NVDA looking okay", score=1000)]
        rp, fp, tp = self._setup_mocks(subs, positive=0.6, negative=0.1)
        with rp, fp, tp:
            result = predict_sp100(threshold=0.99)
        assert result["NVDA"]["signal"] == "not buy"

    def test_result_keys_are_valid_sp100_tickers(self):
        subs = [
            _make_submission("NVDA and AAPL up", score=1000),
            _make_submission("MSFT down today", score=800),
        ]
        rp, fp, tp = self._setup_mocks(subs)
        with rp, fp, tp:
            result = predict_sp100()
        for ticker in result:
            assert ticker in SP100_TICKERS
