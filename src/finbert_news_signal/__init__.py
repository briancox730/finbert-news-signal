"""finbert-news-signal: a point-in-time NLP news-sentiment signal pipeline.

Stages (each importable on its own):

* ``fetch``    -- optional live Alpaca News/Market-Data pulls (offline runs skip this).
* ``score``    -- headline -> sentiment in [-1, 1] (lexicon default; FinBERT optional).
* ``label``    -- point-in-time, DST-aware mapping of headlines to actionable trading days.
* ``cache``    -- per-symbol-year Parquet cache; content-addressed scores cache.
* ``evaluate`` -- event study + costed long/short with in-sample/validation/test discipline.

``fixtures`` provides synthetic data so the whole pipeline runs with no network or GPU.
"""
from __future__ import annotations

from .evaluate import (
    event_study,
    long_leg_portfolio,
    long_short_portfolio,
    rolling_betas,
    split_dates,
)
from .label import assign_trading_day, build_items, build_signal, trading_close_utc
from .score import score_items, score_lexicon, score_texts

__version__ = "0.1.0"

__all__ = [
    "assign_trading_day",
    "build_items",
    "build_signal",
    "event_study",
    "long_leg_portfolio",
    "long_short_portfolio",
    "rolling_betas",
    "score_items",
    "score_lexicon",
    "score_texts",
    "split_dates",
    "trading_close_utc",
]
