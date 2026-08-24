"""Parquet cache round-trips for news and scores."""
from __future__ import annotations

import pandas as pd

from finbert_news_signal import cache


def test_news_round_trip_preserves_utc(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2],
        "created_at": pd.to_datetime(
            ["2021-03-01T14:00:00Z", "2021-03-01T21:30:00Z"], utc=True
        ),
        "headline": ["Acme beats", "Bolt misses"],
        "source": ["synthetic", "synthetic"],
        "symbols": ["ACME", "BOLT"],
        "n_symbols": [1, 1],
    })
    cache.write_news(df, tmp_path, "ACME", 2021)
    back = cache.read_news(tmp_path, "ACME", 2021)

    pd.testing.assert_frame_equal(df, back)
    # tz survives the round trip
    assert str(back["created_at"].dt.tz) == "UTC"


def test_scores_round_trip(tmp_path):
    empty = cache.read_scores(tmp_path, "lexicon")
    assert empty.empty and list(empty.columns) == ["hkey", "score"]

    df = pd.DataFrame({"hkey": ["a1", "b2"], "score": [0.5, -0.3]})
    cache.write_scores(df, tmp_path, "lexicon")
    back = cache.read_scores(tmp_path, "lexicon")
    pd.testing.assert_frame_equal(df, back)


def test_cache_paths(tmp_path):
    assert cache.news_cache_path(tmp_path, "ACME", 2021).name == "news_ACME_2021.parquet"
    assert cache.scores_cache_path(tmp_path, "finbert").name == "news_scores_finbert.parquet"
