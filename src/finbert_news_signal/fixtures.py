"""Synthetic fixtures so the whole pipeline runs offline, with no network or credentials.

The headlines are invented, generic, and hand-written (see the template pools below) -- they
are not sampled from any real news provider. ``regenerate`` composes them into a small news
table over a date range that spans the March daylight-saving transition, and builds a matching
price panel with a *planted* sentiment signal (positive-scoring names get a small boost to
their next-day return) plus noise. That planted edge is what makes the offline event study
show a visible, positive spread; on real data there is no such guarantee.

The generated CSVs are committed under ``_fixtures/`` and loaded by ``load_news`` /
``load_prices``; ``regenerate`` only needs to be re-run if you want to change the fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .score import score_lexicon

_FIXTURE_DIR = Path(__file__).resolve().parent / "_fixtures"
_NEWS_CSV = _FIXTURE_DIR / "news_fixture.csv"
_PRICES_CSV = _FIXTURE_DIR / "prices_fixture.csv"

MARKET_SYMBOL = "MKT"

# Invented fake tickers -> invented company names. No real securities are referenced.
COMPANIES: dict[str, str] = {
    "ACME": "Acme Robotics",
    "BOLT": "Bolt Energy",
    "CRUX": "Crux Semiconductors",
    "DYNE": "Dyne Biolabs",
    "ECHO": "Echo Networks",
    "FLUX": "Flux Materials",
    "GALE": "Gale Airlines",
    "HALO": "Halo Health",
    "IRIS": "Iris Optics",
    "JOLT": "Jolt Beverages",
    "KILN": "Kiln Foods",
    "LUMN": "Lumen Softworks",
}

# Hand-written, generic headline templates grouped by intended polarity. ``{co}`` is filled
# with a company name. These are deliberately synthetic and carry no real-world content.
_POSITIVE_TEMPLATES = [
    "{co} tops quarterly profit estimates as revenue growth accelerates",
    "{co} shares surge after upgrade to buy on strong demand",
    "{co} raises full-year guidance; analysts turn bullish",
    "{co} wins major contract, boosting order backlog to a record",
    "{co} announces dividend hike and a new buyback program",
]
_NEGATIVE_TEMPLATES = [
    "{co} misses revenue estimates and warns of weak guidance",
    "{co} shares plunge after a downgrade on margin concerns",
    "{co} cuts outlook as demand declines and an investigation weighs on shares",
    "{co} faces a lawsuit and product recall, with layoffs expected",
    "{co} slumps on a profit warning and rising default risk",
]
_NEUTRAL_TEMPLATES = [
    "{co} to present at an industry conference next week",
    "{co} names a new chief operating officer",
    "{co} schedules its quarterly earnings call date",
    "{co} completes a previously announced facility relocation",
    "{co} publishes its annual sustainability report",
]


def regenerate(seed: int = 7) -> tuple[Path, Path]:
    """Rebuild the committed fixture CSVs deterministically. Returns the two paths written."""
    rng = np.random.default_rng(seed)
    symbols = list(COMPANIES)
    # Business days spanning the 2021-03-14 US DST transition.
    days = pd.bdate_range("2021-02-01", "2021-04-16")
    templates = {
        "pos": _POSITIVE_TEMPLATES,
        "neg": _NEGATIVE_TEMPLATES,
        "neu": _NEUTRAL_TEMPLATES,
    }
    bucket_choices = np.array(["pos", "neg", "neu"])
    bucket_probs = np.array([0.40, 0.35, 0.25])

    news_rows: list[dict] = []
    # planted[sym][day] = mean lexicon score of that name-day's headlines
    planted = pd.DataFrame(0.0, index=days, columns=symbols)
    next_id = 100000
    for day in days:
        n_active = int(rng.integers(8, len(symbols) + 1))
        active = rng.choice(symbols, size=n_active, replace=False)
        for sym in active:
            n_head = int(rng.integers(1, 3))
            headlines = []
            for _ in range(n_head):
                bucket = rng.choice(bucket_choices, p=bucket_probs)
                template = templates[bucket][int(rng.integers(0, 5))]
                headline = template.format(co=COMPANIES[sym])
                # Timestamp inside regular trading hours (09:35-15:30 ET) so the headline
                # maps to this same session's close under the point-in-time rule.
                minute = int(rng.integers(9 * 60 + 35, 15 * 60 + 30))
                stamp_et = pd.Timestamp(day) + pd.Timedelta(minutes=minute)
                stamp_utc = stamp_et.tz_localize("America/New_York").tz_convert("UTC")
                news_rows.append({
                    "id": next_id,
                    "symbol": sym,
                    "created_at": stamp_utc.isoformat(),
                    "headline": headline,
                    "source": "synthetic",
                    "n_symbols": 1,
                })
                next_id += 1
                headlines.append(headline)
            planted.loc[day, sym] = float(np.mean(score_lexicon(headlines)))

    # Prices: market factor + idiosyncratic noise, plus a planted next-day signal boost.
    mkt_ret = rng.normal(0.0002, 0.009, size=len(days))
    betas = {sym: float(rng.uniform(0.8, 1.2)) for sym in symbols}
    price_data: dict[str, np.ndarray] = {}
    alpha = 0.012
    for sym in symbols:
        ret = betas[sym] * mkt_ret + rng.normal(0.0, 0.006, size=len(days))
        signal = planted[sym].to_numpy()
        # signal on day t moves the t -> t+1 return
        ret[1:] += alpha * signal[:-1]
        price_data[sym] = 100.0 * np.cumprod(1.0 + ret)
    mkt_price = 400.0 * np.cumprod(1.0 + mkt_ret)

    news_df = pd.DataFrame(news_rows)
    prices_df = pd.DataFrame(price_data, index=days)
    prices_df[MARKET_SYMBOL] = mkt_price
    prices_df.index.name = "date"

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    news_df.to_csv(_NEWS_CSV, index=False)
    prices_df.to_csv(_PRICES_CSV)
    return _NEWS_CSV, _PRICES_CSV


def load_news() -> dict[str, pd.DataFrame]:
    """Load the fixture news as ``{symbol: DataFrame}`` matching the fetch-stage schema."""
    flat = pd.read_csv(_NEWS_CSV)
    flat["created_at"] = pd.to_datetime(flat["created_at"], utc=True)
    out: dict[str, pd.DataFrame] = {}
    for sym, group in flat.groupby("symbol"):
        out[str(sym)] = group[
            ["id", "created_at", "headline", "source", "n_symbols"]
        ].reset_index(drop=True)
    return out


def load_prices() -> tuple[pd.DataFrame, pd.Series]:
    """Load the fixture prices as ``(panel, market)`` with a tz-naive date index."""
    df = pd.read_csv(_PRICES_CSV, index_col="date", parse_dates=["date"])
    df.index = pd.DatetimeIndex(df.index).normalize()
    market = df[MARKET_SYMBOL]
    panel = df.drop(columns=[MARKET_SYMBOL])
    return panel, market
