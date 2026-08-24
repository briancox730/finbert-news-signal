"""End-to-end offline smoke test: fixtures -> lexicon -> point-in-time signal -> evaluate."""
from __future__ import annotations

from finbert_news_signal import fixtures
from finbert_news_signal.demo import run_offline


def test_fixtures_load():
    news = fixtures.load_news()
    panel, market = fixtures.load_prices()
    assert len(news) == 12                      # 12 synthetic names
    assert not panel.empty and not market.empty
    assert len(panel.index) == len(market.index)


def test_offline_pipeline_recovers_planted_signal():
    out = run_offline()
    assert out["n_headlines"] > 0
    # The fixture plants a positive signal; the point-in-time event study should recover a
    # positive tercile spread on the held-out test block (the plumbing works end-to-end).
    test_es = out["splits"]["test"]["event_study"]
    assert test_es["n_days"] > 0
    assert test_es["spread_pct"] > 0
