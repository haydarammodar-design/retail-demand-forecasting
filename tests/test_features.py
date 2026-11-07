import numpy as np
import pandas as pd

from demandcast.data import classify_demand, validate_panel
from demandcast.features import make_features

H = 28


def test_no_future_leakage(toy_panel):
    """Changing sales AFTER the cutoff must not change any feature inside the forecast window."""
    cutoff = toy_panel.ds.max() - pd.Timedelta(days=H)
    base, feats = make_features(toy_panel, H, enrich=True)

    tampered = toy_panel.copy()
    tampered.loc[tampered.ds > cutoff, "y"] = 999.0
    alt, _ = make_features(tampered, H, enrich=True)

    window = base.ds > cutoff
    numeric = [f for f in feats if f not in ("unique_id", "event_type_1", "event_name_1")]
    pd.testing.assert_frame_equal(
        base.loc[window, numeric].reset_index(drop=True),
        alt.loc[window, numeric].reset_index(drop=True),
    )


def test_history_only_has_no_external_signals(toy_panel):
    _, feats = make_features(toy_panel, H, enrich=False)
    for f in ["sell_price", "snap", "is_event", "event_type_1"]:
        assert f not in feats


def test_lag_alignment(toy_panel):
    df, _ = make_features(toy_panel, H)
    one = df[df.unique_id == "SKU_0"].reset_index(drop=True)
    assert one.loc[100, f"lag_{H}"] == one.loc[100 - H, "y"]


def test_validate_panel_catches_duplicates(toy_panel):
    validate_panel(toy_panel)
    dup = pd.concat([toy_panel, toy_panel.iloc[:1]])
    try:
        validate_panel(dup)
    except AssertionError:
        return
    raise AssertionError("duplicate rows were not detected")


def test_demand_classification():
    smooth = pd.DataFrame({"unique_id": "a", "y": np.full(100, 5.0)})
    sparse = pd.DataFrame({"unique_id": "b", "y": np.tile([0, 0, 0, 4.0], 25)})
    out = classify_demand(pd.concat([smooth, sparse]))
    assert out.loc["a", "demand_class"] == "smooth"
    assert out.loc["b", "demand_class"] == "intermittent"
