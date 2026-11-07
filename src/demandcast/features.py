"""Feature engineering for the global LightGBM model.

Leakage rule: every feature derived from sales uses a shift of at least ``horizon`` days, so the
model can forecast all ``horizon`` days at once (direct, non-recursive) from data known at the
forecast origin. Prices, calendar, events and SNAP days are treated as known in advance (prices
are set by the retailer; holidays and SNAP schedules are published), as in the M5 competition.
Price statistics use expanding windows so they never look at future prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_FEATURES = ["unique_id", "rmean_7", "rmean_28", "rmean_56", "rmean_112", "rstd_28",
                    "rzero_28", "dow_mean_4w", "dow", "dom", "month", "weekofyear"]
ENRICHED_FEATURES = ["snap", "is_event", "is_closed", "event_type_1", "event_name_1",
                     "days_to_event", "sell_price", "price_rel_item", "price_change_7d",
                     "price_rel_dept", "price_max_ratio"]
CATEGORICAL = ["unique_id", "event_type_1", "event_name_1"]


def make_features(
    panel: pd.DataFrame, horizon: int = 28, enrich: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Return (panel with feature columns, list of feature names to train on).

    enrich=False keeps only sales history + calendar features: used for the "demand signal
    enrichment" ablation that measures how much event / price / SNAP information is worth.
    """
    df = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    h = horizon
    g = df.groupby("unique_id", observed=True)["y"]

    # --- Sales history (all shifted by >= horizon) ---
    lags = [h, h + 7, h + 14, h + 21, h + 28, 364]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    base = df[f"lag_{h}"].groupby(df["unique_id"], observed=True)
    for w in [7, 28, 56, 112]:
        df[f"rmean_{w}"] = base.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    df["rstd_28"] = base.transform(lambda s: s.rolling(28, min_periods=2).std())
    df["rzero_28"] = base.transform(
        lambda s: (s == 0).astype(float).where(s.notna()).rolling(28, min_periods=1).mean()
    )
    # Same-weekday average over the last 4 available weeks: the SKU's weekly profile.
    same_dow = [f"lag_{lag}" for lag in lags[:4] if lag % 7 == 0]
    df["dow_mean_4w"] = df[same_dow].mean(axis=1) if same_dow else df[f"lag_{h}"]

    # --- Calendar: free, deterministic information ---
    df["dow"] = df.ds.dt.dayofweek.astype("int8")
    df["dom"] = df.ds.dt.day.astype("int8")
    df["month"] = df.ds.dt.month.astype("int8")
    df["weekofyear"] = df.ds.dt.isocalendar().week.astype("int8")

    feats = HISTORY_FEATURES + [f"lag_{lag}" for lag in lags]

    if enrich:
        df["is_event"] = (df.event_type_1 != "none").astype("int8")
        df["is_closed"] = df["is_closed"].astype("int8")
        # Days until the next event: shoppers stock up before Thanksgiving, Super Bowl, etc.
        ev_dates = np.sort(df.loc[df.is_event == 1, "ds"].unique())
        idx = np.searchsorted(ev_dates, df.ds.values)
        has_next = idx < len(ev_dates)
        nxt = ev_dates[np.minimum(idx, len(ev_dates) - 1)]
        days = (nxt - df.ds.values) / np.timedelta64(1, "D")
        df["days_to_event"] = np.where(has_next, days, 30).clip(0, 30)
        # Prices: relative to the SKU's own history, week-over-week change, vs. department.
        gp = df.groupby("unique_id", observed=True)["sell_price"]
        df["price_rel_item"] = df.sell_price / gp.transform(lambda s: s.expanding().mean())
        df["price_max_ratio"] = df.sell_price / gp.cummax()  # < 1  => discounted
        df["price_change_7d"] = df.sell_price / gp.shift(7) - 1
        df["price_rel_dept"] = df.sell_price / df.groupby("ds")["sell_price"].transform("mean")
        feats += ENRICHED_FEATURES

    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df, feats
