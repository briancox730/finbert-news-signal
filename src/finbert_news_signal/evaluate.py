"""Evaluation stage: does the signal predict cross-sectional forward returns?

Three lenses, all net of trading cost and all respecting the ``t -> t+1`` realization rule:

* ``event_study`` -- pooled tercile spread on market-adjusted forward returns (top minus
  bottom), with a Newey-West t on the daily spread series.
* ``long_short_portfolio`` -- daily-rebalanced, dollar-neutral (optionally beta-neutral)
  long-top / short-bottom tercile book.
* ``long_leg_portfolio`` -- the retail-executable long-only top tercile, reported with its
  alpha and beta vs the market.

Sample-split discipline (``split_dates``) keeps specification search honest: freeze choices
on the in-sample block, tune on validation, and touch the test block as rarely as possible.
An embargo drops the last ``horizon`` day(s) of each block so a position opened at the end of
one block cannot realize its return inside the next -- the analogue of the point-in-time rule
applied at the split boundary.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .stats import newey_west_tstat, ols_hac

TRADING_DAYS = 252


# --------------------------------------------------------------------------- splits

def split_dates(
    index: pd.DatetimeIndex,
    *,
    is_end: str,
    val_end: str,
    embargo: int = 1,
) -> dict[str, pd.DatetimeIndex]:
    """Partition a date index into in-sample / validation / test blocks with an embargo.

    Args:
        index: the (sorted) trading dates to split.
        is_end: last date (inclusive) of the in-sample block.
        val_end: last date (inclusive) of the validation block.
        embargo: number of trailing dates trimmed from each block so a position opened at a
            block's edge cannot realize into the next block (leakage across the boundary).

    Returns:
        ``{"is": ..., "val": ..., "test": ...}`` of disjoint DatetimeIndex blocks.
    """
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(index).normalize().unique()))
    is_cut = pd.Timestamp(is_end).normalize()
    val_cut = pd.Timestamp(val_end).normalize()

    is_block = idx[idx <= is_cut]
    val_block = idx[(idx > is_cut) & (idx <= val_cut)]
    test_block = idx[idx > val_cut]

    def _embargo(block: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return block[:-embargo] if embargo > 0 and len(block) > embargo else (
            block[:0] if embargo > 0 else block
        )

    return {"is": _embargo(is_block), "val": _embargo(val_block), "test": _embargo(test_block)}


# --------------------------------------------------------------------------- returns / betas

def forward_returns(panel: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """``horizon``-day forward simple return, indexed at the entry day ``t``."""
    return panel.shift(-horizon) / panel - 1.0


def rolling_betas(panel: pd.DataFrame, market: pd.Series, *, window: int = 60) -> pd.DataFrame:
    """Rolling market beta per name (cov of daily returns with the market over ``window``)."""
    ret = panel.pct_change()
    mret = market.pct_change()
    var = mret.rolling(window).var()
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for col in panel.columns:
        out[col] = ret[col].rolling(window).cov(mret) / var
    return out


# --------------------------------------------------------------------------- event study

def event_study(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    market: pd.Series,
    *,
    horizon: int = 1,
    min_names: int = 9,
) -> dict:
    """Pooled tercile event study on market-adjusted forward returns.

    Each day, names with a signal are split into terciles; the daily spread is
    ``mean(top market-adjusted fwd) - mean(bottom market-adjusted fwd)``. Reports pooled
    means and a Newey-West t on the daily spread series.
    """
    fwd = forward_returns(panel, horizon=horizon)
    mfwd = forward_returns(market.to_frame("m"), horizon=horizon)["m"]
    madj = fwd.sub(mfwd, axis=0)

    spreads, tops, bots, counts = [], [], [], []
    n_top_obs = n_bot_obs = 0
    for day in signal.index:
        s = signal.loc[day].dropna()
        s = s[s.index.isin(madj.columns)]
        r = madj.loc[day, s.index]
        keep = s.index[r.notna().to_numpy()]
        s, r = s.loc[keep], r.loc[keep]
        if len(s) < min_names:
            continue
        k = len(s) // 3
        if k < 1:
            continue
        order = s.sort_values()
        bot_i = order.index[:k]
        top_i = order.index[-k:]
        top_mean, bot_mean = r[top_i].mean(), r[bot_i].mean()
        spreads.append(top_mean - bot_mean)
        tops.append(top_mean)
        bots.append(bot_mean)
        counts.append(len(s))
        n_top_obs += k
        n_bot_obs += k

    spreads_arr = np.asarray(spreads, dtype=float)
    return {
        "n_days": len(spreads),
        "avg_names": float(np.mean(counts)) if counts else float("nan"),
        "top_mean_pct": float(np.mean(tops) * 100) if tops else float("nan"),
        "bot_mean_pct": float(np.mean(bots) * 100) if bots else float("nan"),
        "spread_pct": float(np.mean(spreads_arr) * 100) if spreads else float("nan"),
        "nw_t": newey_west_tstat(spreads_arr),
        "n_top_obs": n_top_obs,
        "n_bot_obs": n_bot_obs,
    }


# --------------------------------------------------------------------------- portfolios

def _tercile_weights(
    signal: pd.DataFrame, cols: Sequence[str], *, min_names: int, long_only: bool
) -> tuple[pd.DataFrame, list]:
    """Daily weight matrix. Long-only: equal-weight top tercile (gross 1). Otherwise
    ``+1/k`` on the top tercile and ``-1/k`` on the bottom (dollar-neutral)."""
    weights = pd.DataFrame(0.0, index=signal.index, columns=list(cols))
    active = []
    for day in signal.index:
        s = signal.loc[day].dropna()
        s = s[s.index.isin(cols)]
        if len(s) < min_names:
            continue
        k = len(s) // 3
        if k < 1:
            continue
        order = s.sort_values()
        weights.loc[day, order.index[-k:]] = 1.0 / k
        if not long_only:
            weights.loc[day, order.index[:k]] = -1.0 / k
        active.append(day)
    return weights, active


def _performance(net: pd.Series, turnover: pd.Series) -> dict:
    net = net.dropna()
    if len(net) < 8:
        return {
            "n": len(net),
            "ann_ret_pct": float("nan"),
            "sharpe": float("nan"),
            "nw_t": float("nan"),
            "avg_turnover": float("nan"),
            "win_pct": float("nan"),
        }
    std = net.std()
    return {
        "n": len(net),
        "ann_ret_pct": float(net.mean() * TRADING_DAYS * 100),
        "sharpe": float(net.mean() / std * np.sqrt(TRADING_DAYS)) if std else 0.0,
        "nw_t": newey_west_tstat(net.to_numpy()),
        "avg_turnover": float(turnover.reindex(net.index).mean()),
        "win_pct": float((net > 0).mean() * 100),
    }


def long_short_portfolio(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    market: pd.Series,
    betas: pd.DataFrame | None = None,
    *,
    cost_bps: float = 0.0,
    min_names: int = 9,
    beta_neutral: bool = True,
) -> dict:
    """Daily-rebalanced long-top / short-bottom tercile book, net of cost.

    ``cost_bps`` is charged per unit of one-sided turnover. With ``beta_neutral`` and a
    ``betas`` frame, the residual market exposure of the book is hedged out with the market's
    forward return.
    """
    cols = [c for c in panel.columns if c in signal.columns]
    weights, active = _tercile_weights(signal[cols], cols, min_names=min_names, long_only=False)
    if not active:
        return _performance(pd.Series(dtype=float), pd.Series(dtype=float))

    fwd = panel[cols].pct_change().shift(-1)
    mfwd = market.pct_change().shift(-1)
    weights = weights.loc[active]
    gross = (weights * fwd.reindex(weights.index)).sum(axis=1)
    if beta_neutral and betas is not None:
        net_beta = (weights * betas[cols].reindex(weights.index)).sum(axis=1)
        gross = gross - net_beta * mfwd.reindex(weights.index)

    dweights = weights.diff()
    dweights.iloc[0] = weights.iloc[0]
    turnover = dweights.abs().sum(axis=1)
    net = (gross - turnover * (cost_bps / 1e4)).dropna()
    return _performance(net, turnover)


def long_leg_portfolio(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    market: pd.Series,
    *,
    cost_bps: float = 0.0,
    min_names: int = 9,
) -> dict:
    """Long-only equal-weight top-tercile book, net of cost, with alpha/beta vs the market."""
    cols = [c for c in panel.columns if c in signal.columns]
    weights, active = _tercile_weights(signal[cols], cols, min_names=min_names, long_only=True)
    if not active:
        return _performance(pd.Series(dtype=float), pd.Series(dtype=float))

    fwd = panel[cols].pct_change().shift(-1)
    mfwd = market.pct_change().shift(-1)
    weights = weights.loc[active]
    gross = (weights * fwd.reindex(weights.index)).sum(axis=1)
    dweights = weights.diff()
    dweights.iloc[0] = weights.iloc[0]
    turnover = dweights.abs().sum(axis=1)
    net = (gross - turnover * (cost_bps / 1e4)).dropna()
    perf = _performance(net, turnover)

    aligned = pd.concat(
        [net.rename("y"), mfwd.reindex(net.index).rename("m")], axis=1
    ).dropna()
    if len(aligned) > 20:
        design = np.column_stack([np.ones(len(aligned)), aligned["m"].to_numpy()])
        params, tvalues = ols_hac(aligned["y"].to_numpy(), design)
        perf["alpha_daily_pct"] = float(params[0] * 100)
        perf["alpha_ann_pct"] = float(params[0] * TRADING_DAYS * 100)
        perf["alpha_t"] = float(tvalues[0])
        perf["beta"] = float(params[1])
    return perf
