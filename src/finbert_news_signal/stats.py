"""Small statistics helpers with no dependency beyond NumPy.

The one non-trivial routine is a Newey-West (HAC) t-statistic for the mean of a series. Daily
portfolio/spread returns are autocorrelated, so the ordinary standard error understates
sampling variability; the Bartlett-kernel HAC correction widens it. Implemented directly
rather than pulled from statsmodels to keep the dependency surface minimal.
"""
from __future__ import annotations

import numpy as np


def newey_west_tstat(x: np.ndarray | list[float], lags: int = 10, min_obs: int = 8) -> float:
    """Newey-West HAC t-statistic for H0: mean(x) == 0.

    Regressing ``x`` on a constant, the estimate is the sample mean and its HAC variance is
    ``S / n**2`` where ``S`` is the Bartlett-weighted long-run variance of the residuals::

        S = gamma_0 + 2 * sum_{l=1..L} (1 - l/(L+1)) * gamma_l

    with ``gamma_l`` the lag-``l`` autocovariance sum. ``t = mean / sqrt(S / n**2)``.

    Args:
        x: sample (NaNs are dropped).
        lags: maximum Bartlett lag ``L``; capped at ``n - 1``.
        min_obs: below this many finite observations, return NaN rather than a noisy stat.

    Returns:
        The t-statistic, or NaN when there are too few points or zero variance.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < min_obs:
        return float("nan")

    mean = arr.mean()
    resid = arr - mean
    gamma0 = float(resid @ resid)
    if gamma0 == 0.0:
        return float("nan")

    max_lag = min(lags, n - 1)
    s = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        cov = float(resid[lag:] @ resid[:-lag])
        s += 2.0 * weight * cov

    var_mean = s / (n * n)
    if var_mean <= 0.0:
        return float("nan")
    return float(mean / np.sqrt(var_mean))


def ols_hac(
    y: np.ndarray, x: np.ndarray, lags: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """OLS of ``y`` on design matrix ``x`` with Newey-West (HAC) standard errors.

    ``x`` must already include an intercept column if one is wanted. Returns
    ``(params, tvalues)`` as 1-D arrays aligned to the columns of ``x``. Used for the
    long-leg alpha regression (intercept t vs the market factor).

    The HAC "meat" is the Bartlett-weighted sum of cross products of the score vectors
    ``x_t * e_t``; sandwiching it with ``(X'X)^{-1}`` gives autocorrelation-robust variances.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, k = x.shape

    xtx = x.T @ x
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta

    scores = x * resid[:, None]  # (n, k), row t = x_t * e_t
    meat = scores.T @ scores  # lag-0 term
    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)

    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = np.where(se > 0, beta / se, np.nan)
    return beta, tvalues
