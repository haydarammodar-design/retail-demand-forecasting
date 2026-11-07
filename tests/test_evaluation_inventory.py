import numpy as np
import pandas as pd
import pytest

from demandcast.backtest import fold_cutoffs
from demandcast.evaluation import revenue_weights, score_fold
from demandcast.inventory import reorder_point, safety_stock, simulate_policy, z_value


def test_perfect_forecast_scores_zero(toy_panel):
    cutoff = toy_panel.ds.max() - pd.Timedelta(days=28)
    test = toy_panel[toy_panel.ds > cutoff][["unique_id", "ds", "y"]].assign(model="oracle")
    test["yhat"] = test.y
    s = score_fold(test, toy_panel[toy_panel.ds <= cutoff]).iloc[0]
    assert s.WRMSSE == 0 and s.WAPE == 0 and s.Bias == 0


def test_revenue_weights_sum_to_one(toy_panel):
    assert revenue_weights(toy_panel).sum() == pytest.approx(1.0)


def test_fold_cutoffs_are_ordered_and_spaced():
    cuts = fold_cutoffs(pd.Timestamp("2016-05-22"), 28, 3, 28)
    assert cuts == sorted(cuts)
    assert cuts[-1] == pd.Timestamp("2016-04-24")
    assert (cuts[1] - cuts[0]).days == 28


def test_textbook_safety_stock():
    # z(95%) = 1.645, sigma = 10/day, lead time 4 days -> SS = 1.645 * 10 * 2 = 32.9
    assert z_value(0.95) == pytest.approx(1.645, abs=1e-3)
    assert safety_stock(10, 4, z_value(0.95)) == pytest.approx(32.9, abs=0.01)
    assert reorder_point(20, 10, 4, z_value(0.95)) == pytest.approx(80 + 32.9, abs=0.01)


def test_simulation_perfect_forecast_never_stocks_out():
    rng = np.random.default_rng(1)
    demand = rng.poisson(5, size=(10, 28)).astype(float)
    h = simulate_policy(demand, demand, np.zeros(10), z=0, lead_time=3, review_period=7)
    assert h["lost"].sum() == 0
    np.testing.assert_allclose(h["sold"], demand)


def test_more_safety_stock_means_higher_fill_rate():
    rng = np.random.default_rng(2)
    demand = rng.poisson(5, size=(50, 28)).astype(float)
    flat = np.full_like(demand, 5.0)
    fill = []
    for z in [0.0, 1.0, 2.0]:
        h = simulate_policy(flat, demand, np.full(50, 2.2), z=z, lead_time=3, review_period=7)
        fill.append(h["sold"].sum() / demand.sum())
    assert fill[0] < fill[1] < fill[2]
