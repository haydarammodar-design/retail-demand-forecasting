"""Forecast combination.

M4 and M5 both showed that combining statistical and ML forecasts beats betting on a single
model. Three combiners are built:
  * mean     - equal weights (hard to beat, no estimation error),
  * median   - robust to one member going badly wrong,
  * weighted - inverse-WRMSSE weights estimated ONLY on earlier folds, so fold k's weights never
               see fold k's actuals (no leakage). Fold 1 has no history and falls back to equal.
"""

from __future__ import annotations

import pandas as pd


def combine(fc: pd.DataFrame, members: list[str], fold_scores: pd.DataFrame) -> pd.DataFrame:
    wide = fc[fc.model.isin(members)].pivot_table(
        index=["fold", "unique_id", "ds"], columns="model", values="yhat"
    )[members]
    actual = fc.drop_duplicates(["fold", "unique_id", "ds"]).set_index(
        ["fold", "unique_id", "ds"]
    )["y"]

    out = {
        "Ensemble_mean": wide.mean(axis=1),
        "Ensemble_median": wide.median(axis=1),
    }

    weighted = []
    for fold, part in wide.groupby(level="fold"):
        past = fold_scores[(fold_scores.fold < fold) & fold_scores.model.isin(members)]
        if past.empty:
            wts = pd.Series(1.0, index=members)
        else:
            wts = 1 / past.groupby("model").WRMSSE.mean().reindex(members)
        wts = wts / wts.sum()
        weighted.append(part.mul(wts, axis=1).sum(axis=1))
    out["Ensemble_weighted"] = pd.concat(weighted)

    frames = []
    for name, s in out.items():
        f = s.rename("yhat").to_frame().join(actual).reset_index()
        f["model"] = name
        frames.append(f)
    ens = pd.concat(frames, ignore_index=True)
    ens["cutoff"] = ens.fold.map(fc.drop_duplicates("fold").set_index("fold").cutoff)
    return ens[fc.columns]
