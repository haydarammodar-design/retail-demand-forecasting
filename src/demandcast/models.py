"""Forecasting models (CRISP-DM: Modeling).

Three families, deliberately mixed:
  * Benchmarks       - Naive, SeasonalNaive, 28-day moving average. Any real model must beat these.
  * Statistical      - AutoETS, AutoARIMA (seasonal ARIMA, i.e. SARIMA), and CrostonSBA / ADIDA
                       for intermittent demand. One model fitted per SKU.
  * Machine learning - one *global* LightGBM model trained across all SKUs on lag, rolling,
                       calendar, event and price features (the approach that won M5).
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    ADIDA,
    AutoARIMA,
    AutoETS,
    CrostonSBA,
    Naive,
    SeasonalNaive,
    WindowAverage,
)

BENCHMARKS = ["Naive", "SeasonalNaive", "MovingAverage28"]
STATISTICAL = ["AutoETS", "AutoARIMA", "CrostonSBA", "ADIDA"]


def _stat_models(season_length: int) -> list:
    return [
        Naive(),
        SeasonalNaive(season_length=season_length),
        WindowAverage(window_size=28, alias="MovingAverage28"),
        AutoETS(season_length=season_length),
        # Constrained search space: the default search costs ~4.5 s/SKU on a laptop CPU, this
        # one ~0.5 s/SKU with negligible accuracy loss on daily retail data.
        AutoARIMA(season_length=season_length, max_p=2, max_q=2, max_P=1, max_Q=1,
                  approximation=True),
        CrostonSBA(),
        ADIDA(),
    ]


def forecast_statistical(train: pd.DataFrame, h: int, season_length: int) -> pd.DataFrame:
    """Fit every per-series model on ``train`` and forecast ``h`` days. Long output."""
    sf = StatsForecast(models=_stat_models(season_length), freq="D", n_jobs=-1)
    fc = sf.forecast(df=train[["unique_id", "ds", "y"]], h=h)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    long = fc.melt(id_vars=["unique_id", "ds"], var_name="model", value_name="yhat")
    long["yhat"] = long["yhat"].clip(lower=0)  # demand cannot be negative
    return long


class GlobalLGBM:
    """Direct multi-horizon LightGBM: one model, all SKUs, all horizons at once.

    Works because every sales feature is lagged by >= horizon (see features.py), so the
    features for every day in the forecast window are already known at the cutoff.
    """

    def __init__(self, params: dict, name: str = "LightGBM"):
        self.params = dict(params)
        self.n_estimators = self.params.pop("n_estimators", 500)
        self.name = name
        self.model: lgb.Booster | None = None

    def fit(self, df: pd.DataFrame, feats: list[str]) -> GlobalLGBM:
        df = df.dropna(subset=["rmean_112"])  # need a few months of history per row
        ds = lgb.Dataset(df[feats], label=df["y"], free_raw_data=True)
        self.model = lgb.train(self.params, ds, num_boost_round=self.n_estimators)
        self.feats = feats
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["unique_id", "ds"]].copy()
        out["unique_id"] = out["unique_id"].astype(str)
        out["model"] = self.name
        out["yhat"] = self.model.predict(df[self.feats]).clip(min=0)
        return out

    def importance(self) -> pd.Series:
        imp = self.model.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feats).sort_values(ascending=False)
