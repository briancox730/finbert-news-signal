# finbert-news-signal

An NLP news-sentiment signal pipeline built around **point-in-time
correctness**. It turns a stream of company headlines into a per-name daily
sentiment signal, then checks whether that signal predicts cross-sectional
forward returns, using an event study and a costed long/short portfolio under
strict in-sample / validation / test discipline.

The whole thing runs offline on bundled synthetic fixtures. No network, no API
keys, no GPU. FinBERT and live data are optional upgrades, behind an extra and
a set of environment variables.

- **Default scorer:** a compact finance polarity lexicon, zero heavy
  dependencies.
- **Optional scorer:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert),
  `P(positive) − P(negative)`, behind the `[finbert]` extra (transformers + torch).
- **Leakage handled up front:** every headline is mapped to the first market
  close it could actually be traded into, DST-aware, so no return window ever
  opens before its news.

---

## Pipeline

```
                ┌─────────────────────────────────────────────────────────────┐
                │                    finbert_news_signal                        │
                └─────────────────────────────────────────────────────────────┘

  ALPACA NEWS API                fetch.py            ┌───────────────────────────┐
  (optional, live)  ───────────▶ fetch_news ───────▶ │  cache/  per-symbol-year  │
  ALPACA_API_KEY / SECRET        fetch_prices        │  news_<SYM>_<YEAR>.parquet│
                                                      └────────────┬──────────────┘
  bundled fixtures                                                 │
  (offline default) ──────────▶ fixtures.load_news ───────────────┤
                                 fixtures.load_prices              │
                                                                   ▼
                                                            label.build_items
                                                         (flatten + dedupe reposts)
                                                                   │
                                                                   ▼
                                          score.score_items  ◀── lexicon (default)
                                          headline → [-1, 1]  ◀── FinBERT (optional)
                                                                   │
                                                                   ▼
                                          label.build_signal  (POINT-IN-TIME)
                                    headline @ τ → first close ≥ τ → day t
                                       per (day, symbol) mean score
                                                                   │
                                                                   ▼
                                     evaluate.split_dates  (IS / val / test + embargo)
                                                                   │
                              ┌────────────────────────────────────┼────────────────────────┐
                              ▼                                     ▼                        ▼
                       event_study                       long_short_portfolio       long_leg_portfolio
               tercile spread on market-adj      dollar/beta-neutral, costed     long-only top tercile,
               forward returns, Newey-West t        Newey-West t, turnover        costed, alpha/beta vs mkt
```

Each stage is an importable module with one job: `fetch`, `score`, `label`,
`cache`, `evaluate`, plus `fixtures` for offline data and `stats` for the
Newey-West/HAC helpers.

---

## The point-in-time story

This is the part most pipelines get wrong, and it decides whether the backtest
means anything.

**Filing time vs event time.** A headline has a timestamp `τ`, when it was
published. A naive pipeline pairs a headline with its calendar day's return.
That conflates event time (the day the news is about) with filing time (when
you could actually act). The signal has to be built only from information
available at or before the moment you trade it.

**The rule used here.** Let `close(t)` be the 16:00 America/New_York
regular-session close of trading day `t`. A headline stamped `τ` in the window
`(close(t−1), close(t)]` becomes part of the signal for day `t`. That signal is
acted on at `close(t)` and realized on the `t → t+1` forward return.

**The same-day-leakage trap.** The calendar-day shortcut fails two ways:

1. A headline printed at 09:30 on day `t` would be paired with the
   `close(t−1) → close(t)` return, but part of that move happened before the
   news existed. You would be crediting the signal with a return you could
   never have captured. Mapping the headline forward to `close(t)` and
   realizing on `t → t+1` fixes it.
2. A headline printed at 16:30 on day `t`, after the close, actually belongs to
   the next session. The calendar-day rule still assigns it day `t` and its
   already-finished return, pure look-ahead. Mapping to the first close `≥ τ`
   sends it to day `t+1`, correctly.

`assign_trading_day` implements the rule with a single `searchsorted` for the
first close `≥ τ`; `build_signal` then averages scores per `(day, symbol)`.

**Why DST matters.** The close is a fixed wall-clock time (16:00 ET), so its
UTC instant moves with daylight saving: 21:00 UTC under EST, 20:00 UTC under
EDT. A headline at 20:30 UTC is before the close in winter but after it in
summer. It belongs to different trading days in the two regimes.
`trading_close_utc` does the tz-aware conversion so headlines near the close
are bucketed correctly year-round, and the bundled fixture deliberately spans
the March DST transition.

**Split discipline.** `split_dates` partitions dates into in-sample /
validation / test blocks and applies an embargo: the last `horizon` day(s) of
each block are dropped so a position opened at a block's edge cannot realize
its return inside the next block. Same point-in-time rule, applied at the split
boundary.

---

## Quickstart (offline, no keys, no GPU)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"

# run the full pipeline end-to-end on the synthetic fixtures
finbert-news-demo            # or: python -m finbert_news_signal
```

The demo loads the bundled synthetic news and prices, scores headlines with the
lexicon, builds a point-in-time signal, and reports the event study and costed
portfolios for each split. The fixtures carry a planted signal
(positive-scoring names get a small next-day boost) so the plumbing visibly
works. These numbers demonstrate the pipeline, not a real-world edge.

Use it as a library:

```python
from finbert_news_signal import fixtures, build_items, build_signal, score_items, event_study

news = fixtures.load_news()
panel, market = fixtures.load_prices()

items = build_items(news, start="2021-01-01", end="2021-05-01")
scores = score_items(items, scorer="lexicon")           # or scorer="finbert"
signal, breadth = build_signal(items, scores, panel.index, symbols=list(panel.columns))

print(event_study(signal, panel, market, min_names=6))
```

---

## Enabling FinBERT

The lexicon scorer is the default and needs nothing extra. To use the
transformer model:

```bash
pip install -e ".[finbert]"     # pulls transformers + torch (large)
```

```python
scores = score_items(items, scorer="finbert")   # downloads ProsusAI/finbert on first use
```

`transformers` and `torch` are imported lazily inside `score.score_finbert`, so
importing the package (and the whole lexicon path) never requires them. The
model download happens on first call. CI and the test suite never download it;
the FinBERT wrapper is tested behind a mock.

## Enabling live fetch

```bash
cp .env.example .env
# set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper-trading keys work for market data)
```

```python
from finbert_news_signal import fetch

news = fetch.fetch_news(["ACME", "BOLT"], start_year=2023, end_year=2024)   # cached per year
panel, market = fetch.fetch_prices(["ACME", "BOLT"], "2023-01-01", "2024-12-31")
```

News is cached one `(symbol, year)` Parquet file at a time under `data_cache/`
(override with `FINBERT_NEWS_CACHE`), so re-runs are cheap and only missing
years hit the network.

---

## Development

```bash
pip install -e ".[dev]"
pytest          # offline; never downloads FinBERT
ruff check .
```

The tests pin the point-in-time window assignment (including a DST-transition
case and the same-day-leakage boundary), lexicon scoring sign and monotonicity,
the Parquet cache round-trip, the event-study math against hand-computed
values, the Newey-West statistics, and the split-discipline leakage guard. CI
(GitHub Actions) runs the offline/lexicon path only.

## Project layout

```
src/finbert_news_signal/
  fetch.py       optional live Alpaca News/Market-Data pulls (httpx)
  score.py       lexicon (default) + FinBERT (optional, lazy) scorers
  label.py       point-in-time, DST-aware trading-day assignment + signal build
  cache.py       per-symbol-year Parquet cache + content-addressed scores cache
  evaluate.py    splits, event study, costed long/short & long-leg portfolios
  stats.py       Newey-West t-stat and OLS-HAC helpers (NumPy only)
  fixtures.py    synthetic news + prices generator and loaders
  demo.py        offline end-to-end runner (finbert-news-demo)
tests/           offline test suite
```

## License

MIT. See [LICENSE](LICENSE).
