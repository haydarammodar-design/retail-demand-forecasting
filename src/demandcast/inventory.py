"""Inventory optimization layer (CRISP-DM: Deployment / business translation).

A forecast is only worth what it saves in operations. This module turns forecasts into an
inventory policy and measures the money outcome.

Policy: periodic review, order-up-to (R, S)
  * every R days we review stock and order up to level S,
  * an order arrives L days later (lead time),
  * so each order must protect against demand over the next R + L days:

        S_t      = forecast demand over (R + L)  +  safety stock
        SS       = z * sigma_daily * sqrt(R + L)
        order    = max(0, S_t - inventory position)      (position = on hand + on order)

  z comes from the target cycle service level (z = 1.645 -> 95%), sigma_daily is the SKU's
  forecast error standard deviation measured out-of-sample on earlier backtest folds.
  A better forecast has smaller sigma -> smaller safety stock for the same service.

Simulation uses actual sales from the final backtest fold, lost-sales (no backorders).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def z_value(service_level: float) -> float:
    return float(norm.ppf(service_level))


def safety_stock(sigma_daily, lead_time: float, z: float, review_period: float = 0.0):
    """Classic SS = z * sigma * sqrt(protection period). Works on scalars or arrays."""
    return z * np.asarray(sigma_daily) * np.sqrt(lead_time + review_period)


def reorder_point(mean_daily, sigma_daily, lead_time: float, z: float):
    """Continuous-review reorder point: expected lead-time demand + safety stock."""
    return np.asarray(mean_daily) * lead_time + safety_stock(sigma_daily, lead_time, z)


def residual_sigma(forecasts: pd.DataFrame, folds: list[int]) -> pd.DataFrame:
    """Out-of-sample daily forecast-error std per (model, SKU) from the given folds."""
    f = forecasts[forecasts.fold.isin(folds)]
    sig = (f.yhat - f.y).groupby([f.model, f.unique_id]).std(ddof=1)
    return sig.rename("sigma").reset_index()


def simulate_policy(
    yhat: np.ndarray,
    demand: np.ndarray,
    sigma: np.ndarray,
    z: float,
    lead_time: int,
    review_period: int,
) -> dict[str, np.ndarray]:
    """Simulate an (R, S) policy for many SKUs at once.

    yhat, demand: (n_sku, H) daily forecast and actual demand. sigma: (n_sku,).
    Returns daily arrays of on-hand stock, units sold and units lost.
    """
    n, H = demand.shape
    L, R = lead_time, review_period

    def order_up_to(t: int) -> np.ndarray:
        end = min(t + R + L, H)  # no need to protect beyond the simulated window
        window = end - t
        return yhat[:, t:end].sum(axis=1) + z * sigma * np.sqrt(window)

    on_hand = order_up_to(0)  # start "in steady state": last cycle's order just arrived
    pipeline = np.zeros((n, H + L + 1))  # arrivals scheduled per day
    hist = {k: np.zeros((n, H)) for k in ("on_hand", "sold", "lost", "ordered")}

    for t in range(H):
        on_hand = on_hand + pipeline[:, t]
        if t > 0 and t % R == 0:
            position = on_hand + pipeline[:, t + 1:].sum(axis=1)
            q = np.maximum(0, np.ceil(order_up_to(t) - position))
            pipeline[:, t + L] += q
            hist["ordered"][:, t] = q
        sold = np.minimum(on_hand, demand[:, t])
        on_hand = on_hand - sold
        hist["sold"][:, t] = sold
        hist["lost"][:, t] = demand[:, t] - sold
        hist["on_hand"][:, t] = on_hand
    return hist


def evaluate_policies(
    forecasts: pd.DataFrame,
    prices: pd.Series,
    models: list[str],
    cfg: dict,
) -> pd.DataFrame:
    """Run the policy for every model x service level on the final fold; return KPIs."""
    icfg = cfg["inventory"]
    last = int(forecasts.fold.max())
    sig = residual_sigma(forecasts, [f for f in forecasts.fold.unique() if f < last])
    final = forecasts[forecasts.fold == last]
    days_per_year = 365.0

    rows = []
    for model in models:
        part = final[final.model == model]
        yhat = part.pivot(index="unique_id", columns="ds", values="yhat")
        demand = part.pivot(index="unique_id", columns="ds", values="y").loc[yhat.index]
        s = sig[sig.model == model].set_index("unique_id").sigma.reindex(yhat.index)
        s = s.fillna(s.median()).to_numpy()
        price = prices.reindex(yhat.index).to_numpy()

        for sl in icfg["service_levels"]:
            h = simulate_policy(yhat.to_numpy(), demand.to_numpy(), s, z_value(sl),
                                icfg["lead_time_days"], icfg["review_period_days"])
            inv_value = (h["on_hand"] * price[:, None]).sum(axis=0)  # $ on shelf per day
            holding = inv_value.sum() * icfg["holding_cost_rate_annual"] / days_per_year
            lost_cost = (h["lost"] * price[:, None]).sum() * icfg["gross_margin"]
            rows.append({
                "model": model,
                "service_level": sl,
                "fill_rate": h["sold"].sum() / demand.to_numpy().sum(),
                "stockout_sku_days": float((h["on_hand"] == 0).mean()),
                "avg_inventory_value": inv_value.mean(),
                "avg_units_on_hand": h["on_hand"].sum(axis=0).mean(),
                "holding_cost": holding,
                "lost_margin": lost_cost,
                "total_cost": holding + lost_cost,
            })
    return pd.DataFrame(rows)


def inventory_at_fill_rate(kpis: pd.DataFrame, target: float) -> pd.Series:
    """Interpolate, per model, the average inventory value needed to hit ``target`` fill rate."""
    out = {}
    for model, g in kpis.groupby("model"):
        g = g.sort_values("fill_rate")
        if g.fill_rate.max() < target or g.fill_rate.min() > target:
            out[model] = np.nan
        else:
            out[model] = np.interp(target, g.fill_rate, g.avg_inventory_value)
    return pd.Series(out, name=f"inventory_value_at_{target:.0%}_fill")


def inventory_plan(
    future_fc: pd.DataFrame,
    sigma: pd.Series,
    prices: pd.Series,
    cfg: dict,
    service_level: float,
) -> pd.DataFrame:
    """Per-SKU actionable plan for the next cycle: safety stock, reorder point, order-up-to."""
    icfg = cfg["inventory"]
    L, R = icfg["lead_time_days"], icfg["review_period_days"]
    z = z_value(service_level)
    g = future_fc.sort_values("ds").groupby("unique_id").yhat
    plan = pd.DataFrame({
        "forecast_next_7d": g.apply(lambda s: s.iloc[:7].sum()),
        "forecast_next_28d": g.sum(),
        "mean_daily": g.mean(),
    })
    plan["sigma_daily"] = sigma.reindex(plan.index).fillna(sigma.median())
    plan["safety_stock"] = safety_stock(plan.sigma_daily, L, z, R)
    plan["reorder_point"] = reorder_point(plan.mean_daily, plan.sigma_daily, L, z)
    plan["order_up_to"] = g.apply(lambda s: s.iloc[: R + L].sum()) + plan.safety_stock
    plan["unit_price"] = prices.reindex(plan.index)
    plan["safety_stock_value"] = plan.safety_stock * plan.unit_price
    return plan.round(2).sort_values("safety_stock_value", ascending=False)
