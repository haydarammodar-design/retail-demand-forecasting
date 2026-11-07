"""Static figures for the README and report (matplotlib, PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order, never cycled) + neutral inks.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.axisbelow": True,
})

PRETTY = {
    "Ensemble_mean": "Ensemble (mean)", "Ensemble_median": "Ensemble (median)",
    "Ensemble_weighted": "Ensemble (weighted)",
    "LightGBM_history_only": "LightGBM (sales history only)",
    "LightGBM": "LightGBM (+ events, prices, SNAP)", "MovingAverage28": "Moving average (28d)",
    "SeasonalNaive": "Seasonal naive (last week)",
}


def _save(fig, out: Path) -> None:
    fig.savefig(out)
    plt.close(fig)


def decomposition(panel: pd.DataFrame, out: Path) -> None:
    """STL decomposition of weekly department sales: trend + yearly seasonality + residual.

    Weekly totals remove the day-of-week cycle (shown separately in weekly_profile), which
    makes the slower trend and yearly pattern readable.
    """
    from statsmodels.tsa.seasonal import STL

    daily = panel.groupby("ds").y.sum()
    weekly = daily.resample("W-SUN").sum().iloc[1:-1]  # drop partial first/last weeks
    res = STL(weekly, period=52, seasonal=13).fit()
    parts = [("Observed: weekly units sold", weekly, BLUE), ("Trend", res.trend, BLUE),
             ("Yearly seasonality", res.seasonal, BLUE), ("Residual", res.resid, INK_2)]
    fig, axes = plt.subplots(len(parts), 1, figsize=(10, 7.5), sharex=True)
    for ax, (name, s, color) in zip(axes, parts, strict=False):
        ax.plot(s.index, s.values, color=color)
        ax.set_title(name, fontsize=10)
    axes[-1].axhline(0, color=MUTED, lw=1)
    fig.suptitle("Department sales decomposed with STL (weekly totals)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out)


def weekly_profile(panel: pd.DataFrame, out: Path) -> None:
    daily = panel.groupby("ds").agg(y=("y", "sum"), snap=("snap", "first")).reset_index()
    daily = daily[daily.y > 0]
    daily["dow"] = daily.ds.dt.day_name().str[:3]
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = daily.groupby("dow").y.mean().reindex(order)
    by_snap = daily.groupby("snap").y.mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 1]})
    a1.bar(order, by_dow.values, color=BLUE, width=0.6)
    a1.set_title("Average units sold by weekday")
    a1.set_ylabel("units / day (department)")
    a2.bar(["Regular day", "SNAP day"], by_snap.reindex([0, 1]).values, color=[MUTED, BLUE],
           width=0.6)
    lift = by_snap[1] / by_snap[0] - 1
    a2.set_title(f"SNAP days: {lift:+.0%}")
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out)


def demand_classes(classes: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    c = classes.dropna()
    ax.scatter(c.adi, c.cv2, s=14, color=BLUE, alpha=0.55, edgecolor="none")
    ax.axvline(1.32, color=INK_2, lw=1, ls="--")
    ax.axhline(0.49, color=INK_2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("ADI: average days between sales (log scale)")
    ax.set_ylabel("CV²: variability of sale sizes")
    counts = classes.demand_class.value_counts()
    xmax, ymax = c.adi.max(), c.cv2.quantile(0.99)
    ax.set_ylim(0, ymax * 1.1)
    for label, x, y, ha in [("smooth", 1.29, 0.03, "right"), ("erratic", 1.29, ymax, "right"),
                            ("intermittent", 1.36, 0.03, "left"), ("lumpy", 1.36, ymax, "left")]:
        ax.text(x, y, f"{label}\n{counts.get(label, 0)} SKUs", color=INK, fontsize=8.5, ha=ha,
                va="bottom" if y < 0.1 else "top", fontweight="bold",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_xlim(0.98, xmax * 1.1)
    ax.set_title("Demand patterns (Syntetos-Boylan classification)")
    _save(fig, out)


def model_comparison(summary: pd.DataFrame, out: Path) -> None:
    s = summary.sort_values("WRMSSE", ascending=False)
    labels = [PRETTY.get(m, m) for m in s.index]
    best = s.WRMSSE.idxmin()
    colors = [BLUE if m == best else (INK_2 if m.startswith(("Ensemble", "LightGBM")) else MUTED)
              for m in s.index]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1))
    ax.barh(labels, s.WRMSSE, color=colors, height=0.6)
    for y, v in enumerate(s.WRMSSE):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("WRMSSE, averaged over 3 backtest folds (lower is better)")
    ax.set_xlim(s.WRMSSE.min() * 0.85, s.WRMSSE.max() * 1.06)
    ax.grid(axis="y", visible=False)
    ax.set_title("Forecast accuracy by model")
    _save(fig, out)


def forecast_examples(fc: pd.DataFrame, panel: pd.DataFrame, best: str, skus: list[str],
                      out: Path) -> None:
    last = fc.fold.max()
    f = fc[fc.fold == last]
    start = f.ds.min() - pd.Timedelta(days=42)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for ax, sku in zip(axes.flat, skus, strict=False):
        hist = panel[(panel.unique_id == sku) & (panel.ds >= start)]
        ax.plot(hist.ds, hist.y, color=INK, lw=1.2, label="Actual")
        for model, color in [(best, BLUE), ("SeasonalNaive", ORANGE)]:
            p = f[(f.unique_id == sku) & (f.model == model)]
            ax.plot(p.ds, p.yhat, color=color, label=PRETTY.get(model, model))
        ax.axvline(f.ds.min(), color=MUTED, lw=1, ls="--")
        ax.set_title(sku, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle("Top-revenue SKUs: 28-day forecast vs. actual (dashed line = forecast origin)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out)


def feature_importance(imp: pd.Series, out: Path, top: int = 15) -> None:
    s = (imp / imp.sum()).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s.index, s.values * 100, color=BLUE, height=0.6)
    ax.set_xlabel("share of total gain (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("LightGBM: what drives the forecast")
    _save(fig, out)


def inventory_frontier(kpis: pd.DataFrame, models: list[str], target: float, out: Path) -> None:
    colors = [MUTED, ORANGE, AQUA, BLUE]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, color in zip(models, colors, strict=False):
        g = kpis[kpis.model == model].sort_values("service_level")
        ax.plot(g.avg_inventory_value, g.fill_rate * 100, color=color, marker="o", ms=5,
                label=PRETTY.get(model, model))
    ax.axhline(target * 100, color=INK_2, lw=1, ls="--")
    ax.text(ax.get_xlim()[0], target * 100 + 0.2, f" target fill rate {target:.0%}",
            color=INK_2, fontsize=9, va="bottom")
    ax.set_xlabel("average inventory on shelf ($)")
    ax.set_ylabel("fill rate (% of demand served)")
    ax.set_title("Service vs. inventory investment: better forecasts shift the curve left")
    ax.legend(loc="lower right")
    _save(fig, out)


def fold_stability(fold_scores: pd.DataFrame, models: list[str], out: Path) -> None:
    colors = [MUTED, ORANGE, AQUA, BLUE]
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, color in zip(models, colors, strict=False):
        g = fold_scores[fold_scores.model == model].sort_values("fold")
        ax.plot(g.fold, g.WRMSSE, color=color, marker="o", ms=6, label=PRETTY.get(model, model))
    ax.set_xticks(sorted(fold_scores.fold.unique()))
    ax.set_xlabel("backtest fold (later = more recent)")
    ax.set_ylabel("WRMSSE")
    ax.set_title("Accuracy is consistent across backtest windows")
    ax.legend()
    _save(fig, out)


