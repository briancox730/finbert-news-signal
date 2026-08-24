"""Newey-West t-stat and OLS-HAC helpers, pinned to hand-computed values."""
from __future__ import annotations

import numpy as np

from finbert_news_signal.stats import newey_west_tstat, ols_hac


def test_newey_west_lag0_matches_hand_computation():
    # x = 1..10, lags=0. mean=5.5; gamma0 = sum (x-5.5)^2 = 82.5; var = 82.5/100 = 0.825.
    # t = 5.5 / sqrt(0.825) = 6.05530...
    x = np.arange(1, 11, dtype=float)
    t = newey_west_tstat(x, lags=0)
    assert abs(t - 6.055300708194983) < 1e-9


def test_newey_west_guards():
    # Fewer than min_obs -> NaN.
    assert np.isnan(newey_west_tstat([1.0, 2.0, 3.0]))
    # Zero variance -> NaN (no signal to test).
    assert np.isnan(newey_west_tstat([5.0] * 12))


def test_ols_hac_recovers_exact_line():
    # y = 2 + 3x exactly -> params recovered; residuals ~0.
    x = np.arange(20, dtype=float)
    y = 2.0 + 3.0 * x
    design = np.column_stack([np.ones_like(x), x])
    params, _ = ols_hac(y, design, lags=4)
    assert abs(params[0] - 2.0) < 1e-9
    assert abs(params[1] - 3.0) < 1e-9


def test_ols_hac_mean_only_matches_newey_west():
    # OLS on a constant column must reproduce the mean-only NW t-stat (cross-check the two).
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1.0, size=50)
    ones = np.ones((len(x), 1))
    _, tvalues = ols_hac(x, ones, lags=10)
    assert abs(tvalues[0] - newey_west_tstat(x, lags=10)) < 1e-9
