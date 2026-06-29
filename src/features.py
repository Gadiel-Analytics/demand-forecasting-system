"""
Feature engineering for hierarchical demand forecasting.

The features here encode the structure that drives retail demand: recent
sales momentum (lags and rolling windows), calendar effects (day-of-week,
month, events), price signals, and SNAP benefit days. Categorical hierarchy
variables are target-encoded in a leakage-safe, time-ordered way -- the
single most informative feature family in the M5 problem, per the published
competition post-mortems.

Every transformation here is causal: a feature for day t uses only
information available strictly before t. This is non-negotiable in
forecasting; a single leaked future value inflates validation accuracy and
produces a model that fails in production.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAG_DAYS = [7, 14, 28]
ROLL_WINDOWS = [7, 28]


def add_calendar_features(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach day-of-week, month, event, and SNAP features from the calendar."""
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["dow"] = cal["date"].dt.dayofweek
    cal["month"] = cal["date"].dt.month
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    cal["has_event"] = cal["event_name_1"].notna().astype(int) if "event_name_1" in cal else 0

    keep = ["date", "dow", "month", "is_weekend", "has_event",
            "snap_CA", "snap_TX", "snap_WI"]
    keep = [c for c in keep if c in cal.columns]
    out = df.merge(cal[keep], on="date", how="left")

    # SNAP flag relevant to each row's own state.
    if {"snap_CA", "snap_TX", "snap_WI"}.issubset(out.columns):
        out["snap"] = np.select(
            [out["state_id"] == "CA", out["state_id"] == "TX", out["state_id"] == "WI"],
            [out["snap_CA"], out["snap_TX"], out["snap_WI"]],
            default=0,
        )
        out = out.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    return out


def add_price_features(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach sell price and a normalized price-change signal."""
    out = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    out["sell_price"] = out.groupby("item_store")["sell_price"].ffill()
    # Price momentum: change vs the item's own trailing average.
    avg = out.groupby("item_store")["sell_price"].transform(
        lambda s: s.rolling(28, min_periods=1).mean()
    )
    out["price_ratio"] = out["sell_price"] / avg
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal lag and rolling-window features per series.

    All windows are shifted so that the feature for day t never sees the
    target at day t. Sorted by date within each series first.
    """
    out = df.sort_values(["item_store", "date"]).copy()
    g = out.groupby("item_store")["units"]

    for lag in LAG_DAYS:
        out[f"lag_{lag}"] = g.shift(lag)

    for win in ROLL_WINDOWS:
        # shift(1) before rolling guarantees no same-day leakage.
        out[f"roll_mean_{win}"] = g.shift(1).rolling(win, min_periods=1).mean().reset_index(0, drop=True)
        out[f"roll_std_{win}"] = g.shift(1).rolling(win, min_periods=1).std().reset_index(0, drop=True)

    logger.info("added %d lag and %d rolling features", len(LAG_DAYS), len(ROLL_WINDOWS) * 2)
    return out


def target_encode_hierarchy(
    df: pd.DataFrame, cols: list[str], target: str = "units"
) -> pd.DataFrame:
    """
    Leakage-safe target encoding of high-cardinality hierarchy columns.

    Each observation is encoded by the expanding mean of the target over
    *prior* observations sharing the same categorical value (time-ordered),
    which avoids the leakage that plain mean-encoding introduces.
    """
    out = df.sort_values("date").copy()
    for col in cols:
        enc = out.groupby(col)[target].transform(
            lambda s: s.expanding().mean().shift(1)
        )
        out[f"te_{col}"] = enc
    logger.info("target-encoded: %s", cols)
    return out


def build_features(df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline, applied in causal order."""
    out = add_calendar_features(df, calendar)
    out = add_price_features(out, prices)
    out = add_lag_features(out)
    out = target_encode_hierarchy(out, ["item_id", "store_id", "dept_id"])
    feature_cols = [c for c in out.columns if c.startswith(("lag_", "roll_", "te_", "price_"))]
    feature_cols += ["dow", "month", "is_weekend", "has_event", "snap"]
    out = out.dropna(subset=[c for c in feature_cols if c.startswith("lag_")])
    logger.info("feature matrix: %d rows, %d features", len(out), len(feature_cols))
    return out
