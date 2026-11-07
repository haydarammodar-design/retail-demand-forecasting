"""End-to-end pipeline, one CLI stage per CRISP-DM phase.

    demandcast prepare     # data understanding + preparation   -> data/processed/panel.parquet
    demandcast eda         # exploratory figures                  -> reports/figures/eda_*.png
    demandcast backtest    # modeling + evaluation                -> reports/metrics.csv
    demandcast inventory   # business evaluation of forecasts     -> reports/inventory_kpis.csv
    demandcast forecast    # deployment: next-28-day forecast     -> reports/inventory_plan.csv
    demandcast all         # everything, in order
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

import pandas as pd

from . import plots
from .backtest import run_backtest
from .config import load_config
from .data import classify_demand, future_frame, load_panel, prepare
from .evaluation import revenue_weights, summarize
from .features import make_features
from .inventory import evaluate_policies, inventory_at_fill_rate, inventory_plan, residual_sigma
from .models import GlobalLGBM, forecast_statistical

log = logging.getLogger("demandcast")
CANDIDATES = ["Ensemble_mean", "Ensemble_median", "Ensemble_weighted", "LightGBM",
              "AutoETS", "AutoARIMA"]


def stage_prepare(cfg):
    prepare(cfg)


def stage_eda(cfg):
    panel = load_panel(cfg)
    fig = cfg["output"]["figures_dir"]
    classes = classify_demand(panel)
    classes.to_csv(cfg["output"]["reports_dir"] / "demand_classes.csv")
    plots.decomposition(panel, fig / "eda_decomposition.png")
    plots.weekly_profile(panel, fig / "eda_weekly_snap.png")
    plots.demand_classes(classes, fig / "eda_demand_classes.png")


def stage_backtest(cfg):
    panel = load_panel(cfg)
    rep = cfg["output"]["reports_dir"]
    res = run_backtest(cfg, panel)
    res["forecasts"].to_parquet(rep / "forecasts.parquet", index=False)
    res["fold_scores"].to_csv(rep / "fold_scores.csv", index=False)
    res["importance"].to_csv(rep / "feature_importance.csv", header=["gain"])
    summary = summarize(res["fold_scores"])
    summary.to_csv(rep / "metrics.csv")
    log.info("\n%s", summary.round(3).to_string())
    _backtest_figures(cfg, panel, res["forecasts"], res["fold_scores"], summary,
                      res["importance"])


def _backtest_figures(cfg, panel, fc, fold_scores, summary, importance):
    fig = cfg["output"]["figures_dir"]
    best = summary.loc[summary.index.isin(CANDIDATES)].index[0]
    plots.model_comparison(summary, fig / "model_comparison.png")
    plots.feature_importance(importance, fig / "feature_importance.png")
    last_cut = fc.cutoff.max()
    top = revenue_weights(panel[panel.ds <= last_cut]).nlargest(4).index.tolist()
    plots.forecast_examples(fc, panel, best, top, fig / "forecast_examples.png")
    plots.fold_stability(fold_scores, ["MovingAverage28", "SeasonalNaive", "AutoETS", best],
                         fig / "fold_stability.png")


def stage_inventory(cfg):
    panel = load_panel(cfg)
    rep = cfg["output"]["reports_dir"]
    fc = pd.read_parquet(rep / "forecasts.parquet")
    summary = pd.read_csv(rep / "metrics.csv", index_col=0)
    best = summary.loc[summary.index.isin(CANDIDATES)].index[0]
    models = ["MovingAverage28", "SeasonalNaive", "AutoETS", best]

    last_cut = fc.cutoff.max()
    prices = panel[panel.ds <= last_cut].groupby("unique_id").sell_price.last()
    kpis = evaluate_policies(fc, prices, models, cfg)
    kpis.to_csv(rep / "inventory_kpis.csv", index=False)

    target = cfg["inventory"]["target_fill_rate"]
    inv = inventory_at_fill_rate(kpis, target)
    base = inv.get("MovingAverage28")
    headline = pd.DataFrame({"inventory_needed": inv,
                             "saving_vs_moving_average": 1 - inv / base})
    headline.to_csv(rep / "inventory_headline.csv")
    log.info("\n%s", headline.round(3).to_string())
    plots.inventory_frontier(kpis, models, target, cfg["output"]["figures_dir"] /
                             "inventory_frontier.png")


def stage_forecast(cfg):
    """Retrain the winning approach on ALL history and forecast the next horizon."""
    panel = load_panel(cfg)
    rep, mcfg = cfg["output"]["reports_dir"], cfg["models"]
    h = cfg["backtest"]["horizon"]
    summary = pd.read_csv(rep / "metrics.csv", index_col=0)
    fold_scores = pd.read_csv(rep / "fold_scores.csv")
    best = summary.loc[summary.index.isin(CANDIDATES)].index[0]
    members = mcfg["ensemble_members"] if best.startswith("Ensemble") else [best]

    fut = future_frame(cfg, panel, h)
    preds = []
    stat_needed = [m for m in members if m != "LightGBM"]
    if stat_needed:
        start = panel.ds.max() - pd.Timedelta(days=mcfg["stats_train_days"] - 1)
        st = forecast_statistical(panel[panel.ds >= start], h, mcfg["season_length"])
        preds.append(st[st.model.isin(stat_needed)])
    if "LightGBM" in members:
        full, feats = make_features(pd.concat([panel, fut], ignore_index=True), h, enrich=True)
        m = GlobalLGBM(mcfg["lightgbm"]).fit(full[full.y.notna()], feats)
        preds.append(m.predict(full[full.y.isna()]))
    wide = pd.concat(preds).pivot_table(index=["unique_id", "ds"], columns="model",
                                        values="yhat")[members]
    if best == "Ensemble_weighted":
        w = 1 / fold_scores[fold_scores.model.isin(members)].groupby("model").WRMSSE.mean()
        yhat = wide.mul(w / w.sum(), axis=1).sum(axis=1)
    elif best == "Ensemble_median":
        yhat = wide.median(axis=1)
    else:
        yhat = wide.mean(axis=1)
    future_fc = yhat.rename("yhat").reset_index()
    future_fc.to_csv(rep / "forecast_next_28d.csv", index=False)

    fc_hist = pd.read_parquet(rep / "forecasts.parquet")
    sigma = residual_sigma(fc_hist, list(fc_hist.fold.unique()))
    sigma = sigma[sigma.model == best].set_index("unique_id").sigma
    prices = fut.groupby("unique_id").sell_price.first()
    plan = inventory_plan(future_fc, sigma, prices, cfg,
                          service_level=cfg["inventory"]["target_fill_rate"])
    plan.to_csv(rep / "inventory_plan.csv")
    with open(rep / "forecast_meta.json", "w") as f:
        json.dump({"model": best, "members": members,
                   "from": str(future_fc.ds.min().date()), "to": str(future_fc.ds.max().date()),
                   "total_units": float(future_fc.yhat.sum()),
                   "total_safety_stock_value": float(plan.safety_stock_value.sum())}, f, indent=2)
    log.info("Forecast with %s written; plan for %d SKUs", best, len(plan))


STAGES = {"prepare": stage_prepare, "eda": stage_eda, "backtest": stage_backtest,
          "inventory": stage_inventory, "forecast": stage_forecast}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")  # also silence statsforecast's workers
    cfg = load_config(args.config)
    for name in (STAGES if args.stage == "all" else [args.stage]):
        log.info("=== %s ===", name)
        STAGES[name](cfg)


if __name__ == "__main__":
    main()
