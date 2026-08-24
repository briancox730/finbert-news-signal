"""Event-study math against hand-computed values, and split-discipline leakage guards."""
from __future__ import annotations

import numpy as np
import pandas as pd

from finbert_news_signal.evaluate import event_study, split_dates


def _known_scenario():
    """3 names x 3 days with a flat market, hand-computed forward returns.

    Day D0 signals A>B>C; fwd A=+2%, B=0, C=-1%  -> top=A, bot=C, spread=3%.
    Day D1 signals B>C>A; fwd B=+1%, A=0, C=0    -> top=B, bot=A, spread=1%.
    """
    days = pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-03"])
    panel = pd.DataFrame(
        {
            "A": [100.0, 102.0, 102.0],   # D0->D1 +2%, D1->D2 0%
            "B": [100.0, 100.0, 101.0],   # D0->D1  0%, D1->D2 +1%
            "C": [100.0, 99.0, 99.0],     # D0->D1 -1%, D1->D2 0%
        },
        index=days,
    )
    market = pd.Series([100.0, 100.0, 100.0], index=days)  # flat -> madj == fwd
    signal = pd.DataFrame(
        {
            "A": [0.9, -0.9, np.nan],
            "B": [0.0, 0.9, np.nan],
            "C": [-0.9, 0.0, np.nan],
        },
        index=days,
    )
    return signal, panel, market


def test_event_study_known_values():
    signal, panel, market = _known_scenario()
    res = event_study(signal, panel, market, horizon=1, min_names=3)

    assert res["n_days"] == 2
    assert res["avg_names"] == 3.0
    assert res["n_top_obs"] == 2
    assert res["n_bot_obs"] == 2
    # spread = mean(3%, 1%) = 2%; tops = mean(2%,1%) = 1.5%; bots = mean(-1%,0%) = -0.5%
    assert abs(res["spread_pct"] - 2.0) < 1e-9
    assert abs(res["top_mean_pct"] - 1.5) < 1e-9
    assert abs(res["bot_mean_pct"] - (-0.5)) < 1e-9
    # 2 days < min_obs for the t-stat -> NaN, by design (not a noisy stat).
    assert np.isnan(res["nw_t"])


def test_event_study_respects_min_names():
    signal, panel, market = _known_scenario()
    # Require 4 names but only 3 exist -> no qualifying days.
    res = event_study(signal, panel, market, horizon=1, min_names=4)
    assert res["n_days"] == 0
    assert np.isnan(res["spread_pct"])


def test_split_dates_disjoint_and_ordered():
    idx = pd.bdate_range("2021-01-01", "2021-06-30")
    splits = split_dates(idx, is_end="2021-03-01", val_end="2021-05-01", embargo=1)
    is_b, val_b, test_b = splits["is"], splits["val"], splits["test"]

    # non-empty
    assert len(is_b) and len(val_b) and len(test_b)
    # pairwise disjoint
    assert set(is_b).isdisjoint(val_b)
    assert set(val_b).isdisjoint(test_b)
    assert set(is_b).isdisjoint(test_b)
    # strictly ordered
    assert is_b.max() < val_b.min()
    assert val_b.max() < test_b.min()


def test_split_embargo_blocks_forward_leakage():
    # A position opened on the last train date realizes on the NEXT trading day (horizon=1).
    # With an embargo, that realization day must not fall in the val or test block.
    idx = pd.bdate_range("2021-01-01", "2021-06-30")
    splits = split_dates(idx, is_end="2021-03-01", val_end="2021-05-01", embargo=1)
    full = pd.DatetimeIndex(sorted(idx.normalize().unique()))

    for train_key, future_key in (("is", "val"), ("val", "test")):
        last_train = splits[train_key].max()
        realize_day = full[full > last_train][0]  # t -> t+1 realization
        assert realize_day not in set(splits[future_key])
        assert realize_day not in set(splits["test"])


def test_split_no_embargo_is_contiguous():
    idx = pd.bdate_range("2021-01-01", "2021-06-30")
    splits = split_dates(idx, is_end="2021-03-01", val_end="2021-05-01", embargo=0)
    total = len(splits["is"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(idx)  # nothing dropped when embargo is off
