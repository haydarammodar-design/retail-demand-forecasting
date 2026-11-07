"""Synthetic data so tests run in seconds without downloading M5."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=500, freq="D")
    frames = []
    for i in range(5):
        weekly = 1 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        y = rng.poisson(lam=(i + 1) * weekly).astype("float32")
        frames.append(pd.DataFrame({
            "unique_id": f"SKU_{i}",
            "y": y,
            "ds": dates,
            "wm_yr_wk": 0,
            "event_name_1": np.where(dates.dayofyear == 50, "SuperBowl", "none"),
            "event_type_1": np.where(dates.dayofyear == 50, "Sporting", "none"),
            "event_type_2": "none",
            "snap": (dates.day <= 10).astype("int8"),
            "sell_price": np.float32(2.0 + i),
            "is_closed": False,
        }))
    return pd.concat(frames, ignore_index=True)
