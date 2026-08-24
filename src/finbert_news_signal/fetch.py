"""Fetch stage: optional live pulls from the Alpaca Market Data API.

This is the only stage that touches the network, and it is entirely optional -- the pipeline
runs end to end from bundled fixtures with no credentials. When credentials are present,
``fetch_news`` pulls Benzinga-sourced headlines and caches them one ``(symbol, year)`` Parquet
at a time (see ``cache``); ``fetch_prices`` pulls daily adjusted bars for the return panel.

Only the standard-library-plus-httpx stack is used; the heavyweight vendor SDK is not a
dependency. The News API filters on ``updated_at`` and paginates via ``next_page_token``; we
key everything downstream off the immutable news ``id`` and off ``created_at``, so contiguous
yearly windows give complete, non-double-counted coverage.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx
import pandas as pd

from . import cache as cache_mod
from .config import AlpacaCredentials, cache_dir

_NEWS_PATH = "/v1beta1/news"
_BARS_PATH = "/v2/stocks/{symbol}/bars"
_PAGE_LIMIT = 50  # Alpaca News API max page size


def _auth_headers(creds: AlpacaCredentials) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": creds.api_key,
        "APCA-API-SECRET-KEY": creds.secret_key,
    }


def _fetch_symbol_year(
    client: httpx.Client, creds: AlpacaCredentials, symbol: str, year: int
) -> pd.DataFrame:
    """Pull every headline for one ``(symbol, year)``, following pagination to exhaustion."""
    params = {
        "symbols": symbol,
        "start": f"{year}-01-01T00:00:00Z",
        "end": f"{year + 1}-01-01T00:00:00Z",
        "limit": _PAGE_LIMIT,
        "sort": "asc",
        "include_content": "false",
    }
    rows: list[dict] = []
    page_token: str | None = None
    while True:
        page_params = dict(params)
        if page_token:
            page_params["page_token"] = page_token
        resp = client.get(
            _NEWS_PATH, params=page_params, headers=_auth_headers(creds), timeout=30.0
        )
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("news", []):
            syms = list(item.get("symbols") or [])
            rows.append({
                "id": int(item["id"]),
                "created_at": pd.Timestamp(item["created_at"]).tz_convert("UTC"),
                "headline": (item.get("headline") or "").strip(),
                "source": item.get("source") or "",
                "symbols": "|".join(syms),
                "n_symbols": len(syms),
            })
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return pd.DataFrame(rows, columns=cache_mod.NEWS_COLUMNS)


def fetch_news(
    symbols: list[str],
    start_year: int,
    end_year: int,
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
    sleep: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Fetch (and incrementally cache) headlines per symbol over ``[start_year, end_year]``.

    Returns ``{symbol: DataFrame}`` deduped by news id. Cached symbol-years are read from
    Parquet, so re-runs are cheap and only missing years hit the network.

    Requires Alpaca credentials in the environment (see ``.env.example``). For offline work,
    use ``finbert_news_signal.fixtures`` instead.
    """
    creds = AlpacaCredentials.from_env()
    cdir = Path(cache_path) if cache_path is not None else cache_dir()
    out: dict[str, pd.DataFrame] = {}
    with httpx.Client(base_url=creds.data_url) as client:
        for symbol in symbols:
            parts = []
            for year in range(start_year, end_year + 1):
                path = cache_mod.news_cache_path(cdir, symbol, year)
                if use_cache and path.exists():
                    parts.append(cache_mod.read_news(cdir, symbol, year))
                    continue
                df = _fetch_symbol_year(client, creds, symbol, year)
                cache_mod.write_news(df, cdir, symbol, year)
                parts.append(df)
                time.sleep(sleep)
            combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if not combined.empty:
                combined = combined.drop_duplicates(subset="id").reset_index(drop=True)
                combined["created_at"] = pd.to_datetime(combined["created_at"], utc=True)
            out[symbol] = combined
    return out


def fetch_prices(
    symbols: list[str], start: str, end: str, *, market_symbol: str = "SPY"
) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch daily adjusted closes; returns ``(panel, market)`` with a tz-naive date index.

    ``adjustment=all`` so splits/dividends do not create fake price gaps. Requires Alpaca
    credentials; use ``fixtures`` for the offline path.
    """
    creds = AlpacaCredentials.from_env()
    all_syms = [*symbols, market_symbol]
    closes: dict[str, pd.Series] = {}
    with httpx.Client(base_url=creds.data_url) as client:
        for symbol in all_syms:
            series = _fetch_daily_closes(client, creds, symbol, start, end)
            if not series.empty:
                closes[symbol] = series
    panel = pd.DataFrame({s: closes[s] for s in symbols if s in closes}).sort_index()
    market = closes.get(market_symbol, pd.Series(dtype=float)).reindex(panel.index)
    return panel, market


def _fetch_daily_closes(
    client: httpx.Client, creds: AlpacaCredentials, symbol: str, start: str, end: str
) -> pd.Series:
    params = {
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "adjustment": "all",
        "limit": 10000,
    }
    closes: dict[pd.Timestamp, float] = {}
    page_token: str | None = None
    while True:
        page_params = dict(params)
        if page_token:
            page_params["page_token"] = page_token
        resp = client.get(
            _BARS_PATH.format(symbol=symbol),
            params=page_params,
            headers=_auth_headers(creds),
            timeout=30.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        for bar in payload.get("bars") or []:
            ts = pd.Timestamp(bar["t"]).tz_convert("UTC").tz_localize(None).normalize()
            closes[ts] = float(bar["c"])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return pd.Series(closes, dtype=float).sort_index()
