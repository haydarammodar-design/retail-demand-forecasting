"""Forecast accuracy metrics (CRISP-DM: Evaluation).

Main metric is WRMSSE (Weighted Root Mean Squared Scaled Error), the official M5 metric, here
computed at SKU level:
  * each SKU's RMSE is scaled by the in-sample RMSE of a one-step naive forecast, so slow and
    fast movers are comparable (scale-free),
  * SKUs are weighted by their recent dollar sales, so errors on items that matter to revenue
    count more.
WAPE and bias are reported alongside because business stakeholders read them more easily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def series_scales(train: pd.DataFrame) -> pd.Series:
    """Mean squared one-step naive error per SKU over its training history."""
    d = train.sort_values(["unique_id", "ds"]).groupby("unique_id")["y"].diff()
    scale = (d**2).groupby(train["unique_id"]).mean()
    return scale.replace(0, np.nan)


def revenue_weights(train: pd.DataFrame, last_days: int = 28) -> pd.Series:
    """Share of dollar sales in the last ``last_days`` of training per SKU (sums to 1)."""
    cut = train.ds.max() - pd.Timedelta(days=last_days - 1)
    recent = train[train.ds >= cut]
    rev = (recent.y * recent.sell_price).groupby(recent.unique_id).sum()
    return rev / rev.sum()


def score_fold(fc: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Metrics per model for a single fold. ``fc`` has unique_id, ds, model, yhat, y."""
    scale = series_scales(train)
    w = revenue_weights(train)
    fc = fc.assign(err=fc.yhat - fc.y)

    per_series = fc.groupby(["model", "unique_id"]).agg(
        mse=("err", lambda e: np.mean(e**2)),
        abs_err=("err", lambda e: np.abs(e).sum()),
        sum_err=("err", "sum"),
        sum_y=("y", "sum"),
    ).reset_index()
    per_series["rmsse"] = np.sqrt(per_series.mse / per_series.unique_id.map(scale))
    per_series["w"] = per_series.unique_id.map(w).fillna(0)

    rows = []
    for model, g in per_series.groupby("model"):
        valid = g.rmsse.notna()
        rows.append({
            "model": model,
            "WRMSSE": (g.rmsse[valid] * g.w[valid]).sum() / g.w[valid].sum(),
            "WAPE": g.abs_err.sum() / g.sum_y.sum(),
            "Bias": g.sum_err.sum() / g.sum_y.sum(),
            "RMSE": np.sqrt(g.mse.mean()),
        })
    return pd.DataFrame(rows)


def summarize(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Average metrics over folds, sorted best-first, with skill vs. SeasonalNaive."""
    s = fold_scores.groupby("model")[["WRMSSE", "WAPE", "Bias", "RMSE"]].mean()
    if "SeasonalNaive" in s.index:
        s["Skill_vs_SNaive"] = 1 - s.WRMSSE / s.loc["SeasonalNaive", "WRMSSE"]
    return s.sort_values("WRMSSE")
