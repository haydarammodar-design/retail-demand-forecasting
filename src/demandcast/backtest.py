"""Rolling-origin backtest.

A single train/test split can flatter a model by luck (one unusual month). Instead we slide the
forecast origin back in time ``n_folds`` times and re-train everything at each origin:

    |------------- train -------------| fold 1 (28d) |
    |---------------- train -----------------------| fold 2 (28d) |
    |---------------------- train ---------------------------------| fold 3 (28d) |

All models see exactly the same information at each cutoff, so comparisons are fair.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from .ensemble import combine
from .evaluation import score_fold
from .features import make_features
from .models import GlobalLGBM, forecast_statistical

log = logging.getLogger(__name__)


def fold_cutoffs(last_date: pd.Timestamp, horizon: int, n_folds: int, step: int):
    """Cutoff dates (last day of training) for each fold, oldest first."""
    return [last_date - pd.Timedelta(days=horizon + step * k) for k in reversed(range(n_folds))]


def run_backtest(cfg: dict, panel: pd.DataFrame) -> dict:
    bt, mcfg = cfg["backtest"], cfg["models"]
    h = bt["horizon"]
    cutoffs = fold_cutoffs(panel.ds.max(), h, bt["n_folds"], bt["step"])

    # Features are built once on the full panel. This is leak-free because every sales-derived
    # feature is lagged by >= h days (tested in tests/test_features.py).
    feat_rich, feats_rich = make_features(panel, h, enrich=True)
    feat_hist, feats_hist = make_features(panel, h, enrich=False)

    forecasts, importances = [], None
    for fold, cutoff in enumerate(cutoffs, start=1):
        t0 = time.time()
        test_end = cutoff + pd.Timedelta(days=h)
        log.info("Fold %d: train <= %s, test %s..%s", fold, cutoff.date(),
                 (cutoff + pd.Timedelta(days=1)).date(), test_end.date())

        train = panel[panel.ds <= cutoff]
        stat_start = cutoff - pd.Timedelta(days=mcfg["stats_train_days"] - 1)
        fc_stat = forecast_statistical(train[train.ds >= stat_start], h, mcfg["season_length"])
        log.info("  statistical models done (%.0fs)", time.time() - t0)

        fc_ml = []
        for name, fdf, feats in [
            ("LightGBM", feat_rich, feats_rich),
            ("LightGBM_history_only", feat_hist, feats_hist),
        ]:
            in_train = fdf.ds <= cutoff
            in_test = (fdf.ds > cutoff) & (fdf.ds <= test_end)
            m = GlobalLGBM(mcfg["lightgbm"], name=name).fit(fdf[in_train], feats)
            fc_ml.append(m.predict(fdf[in_test]))
            if name == "LightGBM":
                importances = m.importance()
        log.info("  LightGBM models done (%.0fs)", time.time() - t0)

        fc = pd.concat([fc_stat, *fc_ml], ignore_index=True)
        actual = panel[(panel.ds > cutoff) & (panel.ds <= test_end)][["unique_id", "ds", "y"]]
        fc = fc.merge(actual, on=["unique_id", "ds"], how="inner")
        fc["fold"], fc["cutoff"] = fold, cutoff
        forecasts.append(fc)

    fc = pd.concat(forecasts, ignore_index=True)
    fold_scores = _score_all(fc, panel)

    ens = combine(fc, mcfg["ensemble_members"], fold_scores)
    fc = pd.concat([fc, ens], ignore_index=True)
    fold_scores = _score_all(fc, panel)
    return {"forecasts": fc, "fold_scores": fold_scores, "importance": importances}


def _score_all(fc: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (fold, cutoff), part in fc.groupby(["fold", "cutoff"]):
        s = score_fold(part, panel[panel.ds <= cutoff])
        s["fold"] = fold
        out.append(s)
    return pd.concat(out, ignore_index=True)
