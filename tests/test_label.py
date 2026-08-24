"""Point-in-time labeling: DST-aware closes and the same-day-leakage boundary."""
from __future__ import annotations

import pandas as pd

from finbert_news_signal.label import assign_trading_day, build_signal, trading_close_utc


def _items(created_ats: list[pd.Timestamp]) -> pd.DataFrame:
    """Minimal items frame; assign_trading_day only needs created_at."""
    return pd.DataFrame({
        "sym": ["ACME"] * len(created_ats),
        "id": list(range(len(created_ats))),
        "created_at": pd.to_datetime(pd.Series(created_ats), utc=True),
        "headline": ["x"] * len(created_ats),
        "hnorm": ["x"] * len(created_ats),
        "n_symbols": [1] * len(created_ats),
    })


def test_trading_close_utc_dst_offsets():
    # 16:00 ET is 21:00 UTC under EST (winter) and 20:00 UTC under EDT (summer).
    winter = trading_close_utc(pd.DatetimeIndex(["2021-02-01"]))[0]
    summer = trading_close_utc(pd.DatetimeIndex(["2021-06-01"]))[0]
    assert winter == pd.Timestamp("2021-02-01 21:00", tz="UTC")
    assert summer == pd.Timestamp("2021-06-01 20:00", tz="UTC")


def test_same_day_leakage_boundary():
    # Mon/Tue/Wed all trading days. close(Mon) = 2021-03-01 21:00 UTC (EST).
    cal = pd.DatetimeIndex(["2021-03-01", "2021-03-02", "2021-03-03"])

    def _et(hh, mm):
        return pd.Timestamp(f"2021-03-01 {hh:02d}:{mm:02d}", tz="America/New_York")

    items = _items([
        _et(15, 0),   # before close -> Monday
        _et(16, 0),   # exactly at close -> Monday (window is right-closed)
        _et(16, 30),  # AFTER close -> Tuesday (the leakage trap: not Monday!)
    ])
    out = assign_trading_day(items, cal).sort_values("id")
    days = list(out["day"])
    assert days[0] == pd.Timestamp("2021-03-01")
    assert days[1] == pd.Timestamp("2021-03-01")
    assert days[2] == pd.Timestamp("2021-03-02")


def test_dst_changes_bucketing_for_same_utc_time():
    # A headline at 20:30 UTC is BEFORE the close in winter (close 21:00 UTC) but
    # AFTER it in summer (close 20:00 UTC) -> different trading days.
    winter_cal = pd.DatetimeIndex(["2021-02-01", "2021-02-02"])
    summer_cal = pd.DatetimeIndex(["2021-06-01", "2021-06-02"])

    winter = assign_trading_day(_items([pd.Timestamp("2021-02-01 20:30", tz="UTC")]), winter_cal)
    summer = assign_trading_day(_items([pd.Timestamp("2021-06-01 20:30", tz="UTC")]), summer_cal)

    assert winter["day"].iloc[0] == pd.Timestamp("2021-02-01")   # same day (before close)
    assert summer["day"].iloc[0] == pd.Timestamp("2021-06-02")   # next day (after close)


def test_headline_after_last_close_is_dropped():
    cal = pd.DatetimeIndex(["2021-03-01"])
    # 22:00 UTC is after the only close (21:00 UTC) -> no actionable day.
    out = assign_trading_day(_items([pd.Timestamp("2021-03-01 22:00", tz="UTC")]), cal)
    assert out.empty


def test_build_signal_averages_per_name_day():
    cal = pd.DatetimeIndex(["2021-03-01", "2021-03-02"])
    items = pd.DataFrame({
        "sym": ["ACME", "ACME", "BOLT"],
        "id": [1, 2, 3],
        "created_at": pd.to_datetime(pd.Series([
            pd.Timestamp("2021-03-01 15:00", tz="America/New_York"),
            pd.Timestamp("2021-03-01 15:30", tz="America/New_York"),
            pd.Timestamp("2021-03-01 15:00", tz="America/New_York"),
        ]), utc=True),
        "headline": ["a", "b", "c"],
        "hnorm": ["a", "b", "c"],
        "n_symbols": [1, 1, 1],
    })
    scores = pd.Series([0.4, 0.8, -0.5], index=items.index)
    signal, breadth = build_signal(items, scores, cal, symbols=["ACME", "BOLT"])
    # ACME on Monday = mean(0.4, 0.8) = 0.6; BOLT = -0.5.
    assert abs(signal.loc[pd.Timestamp("2021-03-01"), "ACME"] - 0.6) < 1e-9
    assert abs(signal.loc[pd.Timestamp("2021-03-01"), "BOLT"] - (-0.5)) < 1e-9
    assert breadth.loc[pd.Timestamp("2021-03-01"), "ACME"] == 2
    # Tuesday has no news.
    assert pd.isna(signal.loc[pd.Timestamp("2021-03-02"), "ACME"])
