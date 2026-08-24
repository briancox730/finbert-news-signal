"""Labeling stage: map headlines to the trading day they can first be acted on, then
aggregate to a per-(symbol, day) signal. This is where point-in-time correctness lives.

The rule
--------
Let ``close(t)`` be the 16:00 America/New_York regular-session close of trading day ``t``.
A headline stamped at time ``tau`` in the half-open-then-closed window ``(close(t-1), close(t)]``
becomes part of the signal for day ``t``. That signal is *acted on at* ``close(t)`` and is
realized on the ``t -> t+1`` forward return.

Why the next close and not the calendar day
-------------------------------------------
The tempting shortcut is "assign a headline to its calendar date and use that day's return."
That leaks. A headline printed at 09:30 on day ``t`` would be paired with the ``close(t-1) ->
close(t)`` return, part of which happened *before* the news existed -- you would be scoring a
move you could not have traded. Worse, a headline printed at 16:30 (after the close) belongs
to the *next* session, but the calendar-day rule would still hand it day ``t`` and its already-
finished return. Mapping every headline forward to the first close ``>= tau`` removes both
traps: nothing enters a return window that opened before the news.

Everything is computed in UTC. Because the ET close is a fixed wall-clock time, its UTC
instant shifts with daylight-saving time (21:00 UTC under EST, 20:00 UTC under EDT); the
conversion below is DST-aware so a headline near the close is bucketed correctly year-round.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .text import normalize_headline

MARKET_TZ = "America/New_York"
CLOSE_HOUR = 16  # 16:00 ET regular-session close


def build_items(
    news_by_symbol: dict[str, pd.DataFrame],
    *,
    start: str,
    end: str,
    max_symbols: int = 999,
) -> pd.DataFrame:
    """Flatten per-symbol news into one long table and dedupe reposts.

    Keeps items whose ``created_at`` is in ``[start, end)``. ``max_symbols`` optionally drops
    market-wide wire items (a headline tagging many tickers is rarely company-specific news).
    Reposts are removed by ``(symbol, ET-calendar-day, normalized-headline)``.

    Returns a frame with columns ``sym, id, created_at, headline, hnorm, n_symbols``.
    """
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC")
    cols = ["sym", "id", "created_at", "headline", "hnorm", "n_symbols"]
    frames = []
    for sym, df in news_by_symbol.items():
        if df is None or df.empty:
            continue
        window = df[(df["created_at"] >= lo) & (df["created_at"] < hi)].copy()
        if max_symbols < 999 and "n_symbols" in window.columns:
            window = window[window["n_symbols"] <= max_symbols]
        if window.empty:
            continue
        window["sym"] = sym
        window["hnorm"] = window["headline"].map(normalize_headline)
        day = window["created_at"].dt.tz_convert(MARKET_TZ).dt.normalize().dt.tz_localize(None)
        window["_day"] = day
        window = window.drop_duplicates(subset=["sym", "_day", "hnorm"])
        if "n_symbols" not in window.columns:
            window["n_symbols"] = 1
        frames.append(window[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def trading_close_utc(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each (tz-naive) trading date, its 16:00 ET close as a tz-aware UTC instant.

    DST-aware: the same 16:00 wall-clock close lands on a different UTC hour across the
    spring/fall transitions.
    """
    cal = pd.DatetimeIndex(calendar).normalize()
    local = cal + pd.Timedelta(hours=CLOSE_HOUR)
    localized = local.tz_localize(MARKET_TZ, nonexistent="shift_forward", ambiguous=True)
    return localized.tz_convert("UTC")


def assign_trading_day(items: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach a ``day`` column: the first trading day whose close is ``>= created_at``.

    Headlines after the last close in ``calendar`` have no actionable day and are dropped.
    """
    cal = pd.DatetimeIndex(sorted(pd.DatetimeIndex(calendar).normalize().unique()))
    if items.empty or len(cal) == 0:
        return items.assign(day=pd.Series(dtype="datetime64[ns]"))

    closes = trading_close_utc(cal)
    created = pd.DatetimeIndex(pd.to_datetime(items["created_at"], utc=True))
    # searchsorted(side="left") returns the first index whose close is >= created_at,
    # i.e. the earliest session the news could be traded into at the close.
    pos = closes.searchsorted(created, side="left")
    valid = pos < len(cal)
    out = items.loc[valid].copy()
    out["day"] = cal[pos[valid]]
    return out


def build_signal(
    items: pd.DataFrame,
    scores: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    symbols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate scored headlines into per-(day, symbol) signal and breadth matrices.

    Returns ``(signal, breadth)``: both indexed by the trading ``calendar`` with one column
    per symbol. ``signal`` is the mean headline score for that name-day (NaN when there is no
    news); ``breadth`` is the headline count.
    """
    cal = pd.DatetimeIndex(sorted(pd.DatetimeIndex(calendar).normalize().unique()))
    symbols = list(symbols)
    scored = items.dropna(subset=["hnorm"]).copy()
    scored["score"] = scores.reindex(scored.index).to_numpy()
    scored = scored.dropna(subset=["score"])
    scored = assign_trading_day(scored, cal)
    scored = scored.dropna(subset=["day"])

    if scored.empty:
        empty = pd.DataFrame(index=cal, columns=symbols, dtype=float)
        return empty, empty.copy()

    grp = scored.groupby(["day", "sym"])["score"]
    signal = grp.mean().unstack("sym").reindex(index=cal, columns=symbols)
    breadth = grp.size().unstack("sym").reindex(index=cal, columns=symbols)
    return signal, breadth
