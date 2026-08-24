"""Offline end-to-end demo on the bundled synthetic fixtures.

Run as ``finbert-news-demo`` (console script) or ``python -m finbert_news_signal``. Exercises
every stage -- load news, filter/dedupe, lexicon-score, point-in-time label, evaluate across
in-sample / validation / test blocks -- with zero network access and no heavy dependencies.
"""
from __future__ import annotations

import math

from . import fixtures
from .evaluate import (
    event_study,
    long_leg_portfolio,
    long_short_portfolio,
    rolling_betas,
    split_dates,
)
from .label import build_items, build_signal
from .score import score_items

MIN_NAMES = 6  # the fixture has 12 names; require >=6 with news to form terciles


def run_offline() -> dict:
    """Run the full offline pipeline on fixtures; return the per-split evaluation results."""
    news = fixtures.load_news()
    panel, market = fixtures.load_prices()
    calendar = panel.index

    items = build_items(
        news, start="2021-01-01", end="2021-05-01", max_symbols=999
    )
    scores = score_items(items, scorer="lexicon")
    signal, breadth = build_signal(items, scores, calendar, symbols=list(panel.columns))
    betas = rolling_betas(panel, market, window=20)

    splits = split_dates(calendar, is_end="2021-03-05", val_end="2021-03-26", embargo=1)
    results: dict[str, dict] = {}
    for name, dates in splits.items():
        sig = signal.loc[signal.index.isin(dates)]
        results[name] = {
            "n_signal_days": int(sig.notna().any(axis=1).sum()),
            "event_study": event_study(sig, panel, market, min_names=MIN_NAMES),
            "long_short_5bps": long_short_portfolio(
                sig, panel, market, betas, cost_bps=5.0, min_names=MIN_NAMES
            ),
            "long_leg_5bps": long_leg_portfolio(
                sig, panel, market, cost_bps=5.0, min_names=MIN_NAMES
            ),
        }
    return {"n_headlines": len(items), "breadth_total": int(breadth.sum().sum()), "splits": results}


def _fmt(value: float, spec: str = ".2f") -> str:
    return "nan" if value is None or math.isnan(value) else format(value, spec)


def main() -> None:
    out = run_offline()
    print("=" * 68)
    print("FinBERT news-signal pipeline -- OFFLINE DEMO (synthetic fixtures)")
    print("=" * 68)
    print(f"headlines after dedupe: {out['n_headlines']}   name-day observations: "
          f"{out['breadth_total']}")
    print("scorer: lexicon (default; enable FinBERT with the [finbert] extra)")
    print()
    for name in ("is", "val", "test"):
        res = out["splits"][name]
        es = res["event_study"]
        ls = res["long_short_5bps"]
        ll = res["long_leg_5bps"]
        print(f"[{name.upper():>4}]  signal days={res['n_signal_days']:>3}  "
              f"event-study days={es['n_days']:>3}")
        print(f"        event study : spread={_fmt(es['spread_pct'])}%  "
              f"top={_fmt(es['top_mean_pct'])}%  bot={_fmt(es['bot_mean_pct'])}%  "
              f"NW t={_fmt(es['nw_t'])}")
        print(f"        long/short  : ann={_fmt(ls['ann_ret_pct'])}%  "
              f"sharpe={_fmt(ls['sharpe'])}  NW t={_fmt(ls['nw_t'])}  "
              f"turnover={_fmt(ls['avg_turnover'])}")
        print(f"        long leg    : ann={_fmt(ll['ann_ret_pct'])}%  "
              f"alpha_ann={_fmt(ll.get('alpha_ann_pct', float('nan')))}%  "
              f"beta={_fmt(ll.get('beta', float('nan')))}")
        print()
    print("NOTE: fixtures carry a planted signal so the pipeline visibly works end-to-end.")
    print("      These numbers demonstrate plumbing, not a real-world edge.")


if __name__ == "__main__":
    main()
