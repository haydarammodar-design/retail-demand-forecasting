"""Data acquisition and preparation (CRISP-DM: Data Understanding + Data Preparation).

Turns the wide M5 sales file into a tidy long panel with one row per (SKU, day):

    unique_id | ds | y | sell_price | calendar / event / SNAP columns
"""

from __future__ import annotations

import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Public mirror of the Kaggle M5 files maintained by Nixtla (no Kaggle login needed).
M5_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
M5_FILES = ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"]


def download_m5(raw_dir: Path) -> None:
    """Download and extract the M5 CSVs into ``raw_dir`` if they are not already there."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / f).exists() for f in M5_FILES):
        return
    zip_path = raw_dir / "m5.zip"
    if not zip_path.exists():
        log.info("Downloading M5 data (~50 MB) from %s", M5_URL)
        urllib.request.urlretrieve(M5_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in M5_FILES:
            zf.extract(name, raw_dir)


def load_sales(raw_dir: Path, store_id: str, dept_id: str) -> pd.DataFrame:
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    sales = sales[(sales.store_id == store_id) & (sales.dept_id == dept_id)]
    if sales.empty:
        raise ValueError(f"No rows for store={store_id}, dept={dept_id}")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=["item_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="y",
    )
    return long


def build_panel(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Build the cleaned long-format panel and a data-quality report."""
    dcfg = cfg["data"]
    raw_dir: Path = dcfg["raw_dir"]
    download_m5(raw_dir)

    long = load_sales(raw_dir, dcfg["store_id"], dcfg["dept_id"])
    state = long["state_id"].iloc[0]

    cal = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    cal = cal.rename(columns={"date": "ds", f"snap_{state}": "snap"}).sort_values("ds")
    if "d" not in cal.columns:  # the Nixtla mirror drops the d_1..d_1969 key; rebuild it
        cal["d"] = [f"d_{i}" for i in range(1, len(cal) + 1)]
    cal = cal[["d", "ds", "wm_yr_wk", "event_name_1", "event_type_1", "event_type_2", "snap"]]

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices.store_id == dcfg["store_id"]].drop(columns="store_id")

    df = long.merge(cal, on="d", how="left").merge(prices, on=["item_id", "wm_yr_wk"], how="left")
    df = df.rename(columns={"item_id": "unique_id"})
    df = df.drop(columns=["d", "store_id", "state_id"])
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    report: dict = {"store_id": dcfg["store_id"], "dept_id": dcfg["dept_id"]}
    report["raw_rows"] = len(df)

    # 1) Pre-launch rows: M5 only lists a price once the item is on the shelf. Zeros before that
    #    are "not sold yet", not "zero demand", and would bias every model downward.
    first_price = df[df.sell_price.notna()].groupby("unique_id")["ds"].min()
    df = df.merge(first_price.rename("launch_ds"), on="unique_id", how="left")
    prelaunch = df.ds < df.launch_ds
    report["prelaunch_rows_dropped"] = int(prelaunch.sum())
    df = df[~prelaunch].drop(columns="launch_ds")

    # 2) Restrict history to the study window.
    df = df[df.ds >= pd.Timestamp(dcfg["start_date"])]

    # 3) Gaps in the price file during an item's life -> carry last known price.
    report["missing_prices_filled"] = int(df.sell_price.isna().sum())
    df["sell_price"] = df.groupby("unique_id")["sell_price"].transform(lambda s: s.ffill().bfill())

    # 4) Store closures: Walmart is closed on Christmas, so sales are structurally ~0.
    #    We keep the rows (the calendar is contiguous) but flag them so models can learn it.
    df["is_closed"] = (df.ds.dt.month == 12) & (df.ds.dt.day == 25)
    report["closure_days"] = int(df.loc[df.is_closed, "ds"].nunique())

    # 5) Event columns: NaN means "no event".
    for c in ["event_name_1", "event_type_1", "event_type_2"]:
        df[c] = df[c].fillna("none")

    df["y"] = df["y"].astype("float32")
    df["sell_price"] = df["sell_price"].astype("float32")
    df["snap"] = df["snap"].astype("int8")
    df = df.reset_index(drop=True)

    validate_panel(df)

    report.update(
        n_series=int(df.unique_id.nunique()),
        n_rows=int(len(df)),
        start=str(df.ds.min().date()),
        end=str(df.ds.max().date()),
        pct_zero_sales=round(float((df.y == 0).mean()) * 100, 1),
        mean_daily_units=round(float(df.y.mean()), 2),
    )
    return df, report


def future_frame(cfg: dict, panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rows for the ``horizon`` days after the panel ends, with y unknown (NaN).

    M5 publishes the calendar and planned prices for those days, exactly like a retailer
    knows its own promo plan and holiday calendar before it forecasts.
    """
    raw_dir: Path = cfg["data"]["raw_dir"]
    state = cfg["data"]["store_id"].split("_")[0]
    start = panel.ds.max() + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")

    cal = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    cal = cal.rename(columns={"date": "ds", f"snap_{state}": "snap"})
    cal = cal[cal.ds.isin(dates)][["ds", "wm_yr_wk", "event_name_1", "event_type_1",
                                   "event_type_2", "snap"]]
    if len(cal) < horizon:
        raise ValueError("Calendar does not cover the forecast horizon")

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices.store_id == cfg["data"]["store_id"]].drop(columns="store_id")
    prices = prices.rename(columns={"item_id": "unique_id"})

    ids = pd.DataFrame({"unique_id": panel.unique_id.unique()})
    fut = ids.merge(cal, how="cross").merge(prices, on=["unique_id", "wm_yr_wk"], how="left")
    last_price = panel.sort_values("ds").groupby("unique_id").sell_price.last()
    fut["sell_price"] = fut.sell_price.fillna(fut.unique_id.map(last_price)).astype("float32")
    for c in ["event_name_1", "event_type_1", "event_type_2"]:
        fut[c] = fut[c].fillna("none")
    fut["is_closed"] = (fut.ds.dt.month == 12) & (fut.ds.dt.day == 25)
    fut["snap"] = fut["snap"].astype("int8")
    fut["y"] = np.float32(np.nan)
    return fut[panel.columns]


def validate_panel(df: pd.DataFrame) -> None:
    """Fail loudly on data problems that would silently corrupt a forecast."""
    assert not df.duplicated(["unique_id", "ds"]).any(), "duplicate (unique_id, ds) rows"
    assert (df.y >= 0).all(), "negative sales"
    assert df.sell_price.notna().all(), "missing prices after cleaning"
    gaps = df.groupby("unique_id")["ds"].apply(lambda s: s.diff().dropna().dt.days.max())
    assert (gaps.fillna(1) == 1).all(), "non-contiguous dates within a series"


def classify_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Syntetos-Boylan demand classification (smooth / erratic / intermittent / lumpy).

    ADI = average inter-demand interval, CV2 = squared coefficient of variation of non-zero sizes.
    Cut-offs 1.32 and 0.49 come from Syntetos, Boylan & Croston (2005).
    """

    def _stats(y: pd.Series) -> pd.Series:
        nz = y[y > 0]
        adi = len(y) / max(len(nz), 1)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
        return pd.Series({"adi": adi, "cv2": cv2})

    out = df.groupby("unique_id")["y"].apply(_stats).unstack()
    out["demand_class"] = np.select(
        [
            (out.adi < 1.32) & (out.cv2 < 0.49),
            (out.adi < 1.32) & (out.cv2 >= 0.49),
            (out.adi >= 1.32) & (out.cv2 < 0.49),
        ],
        ["smooth", "erratic", "intermittent"],
        default="lumpy",
    )
    return out


def prepare(cfg: dict) -> pd.DataFrame:
    panel, report = build_panel(cfg)
    out_dir: Path = cfg["data"]["processed_dir"]
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    with open(cfg["output"]["reports_dir"] / "data_quality.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Data quality report: %s", report)
    return panel


def load_panel(cfg: dict) -> pd.DataFrame:
    path = cfg["data"]["processed_dir"] / "panel.parquet"
    if not path.exists():
        return prepare(cfg)
    return pd.read_parquet(path)
