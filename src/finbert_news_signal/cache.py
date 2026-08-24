"""Per-symbol-year Parquet cache for news and a content-addressed cache for scores.

News is fetched and stored one ``(symbol, year)`` Parquet file at a time. That granularity
makes coverage incremental and idempotent: contiguous yearly windows can be re-requested
cheaply, only the missing years hit the network, and re-runs never double-count because
downstream dedup keys on the immutable news id.

The scores cache is keyed by a content hash of the normalized headline, so the (potentially
expensive) FinBERT pass is done once per distinct headline and reused forever.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NEWS_COLUMNS = ["id", "created_at", "headline", "source", "symbols", "n_symbols"]


def news_cache_path(cache_dir: Path, symbol: str, year: int) -> Path:
    return Path(cache_dir) / f"news_{symbol}_{year}.parquet"


def scores_cache_path(cache_dir: Path, scorer: str) -> Path:
    return Path(cache_dir) / f"news_scores_{scorer}.parquet"


def write_news(df: pd.DataFrame, cache_dir: Path, symbol: str, year: int) -> Path:
    """Persist one symbol-year news frame to Parquet, returning the path written."""
    path = news_cache_path(cache_dir, symbol, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_news(cache_dir: Path, symbol: str, year: int) -> pd.DataFrame:
    """Read one symbol-year news frame, restoring a tz-aware UTC ``created_at``."""
    df = pd.read_parquet(news_cache_path(cache_dir, symbol, year))
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def read_scores(cache_dir: Path, scorer: str) -> pd.DataFrame:
    """Return the cached ``(hkey, score)`` table for a scorer (empty frame if none yet)."""
    path = scores_cache_path(cache_dir, scorer)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({"hkey": pd.Series(dtype="object"), "score": pd.Series(dtype="float")})


def write_scores(df: pd.DataFrame, cache_dir: Path, scorer: str) -> Path:
    """Persist the ``(hkey, score)`` table for a scorer."""
    path = scores_cache_path(cache_dir, scorer)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
