"""
Evaluation and the business (financial) translation.

Two layers live here:

1. Forecast accuracy, measured with RMSSE -- the scaled error metric the M5
   competition uses, which is comparable across series of very different
   scales (a metric that treats a 2-unit error on a slow item the same as a
   2-unit error on a fast one would be meaningless across 30,490 series).

2. The financial translation. Accuracy is necessary but not the point. A
   demand-planning system earns its keep by turning the forecast into the
   numbers a planner actually decides on: expected revenue, and the cost of
   getting the forecast wrong in each direction (overstock capital tied up
   vs. stockout sales lost). This layer is what makes the project read as
   business data science rather than a modeling exercise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rmsse(actual: np.ndarray, forecast: np.ndarray, train_actual: np.ndarray) -> float:
    """
    Root Mean Squared Scaled Error.

    The denominator is the in-sample one-step naive error on the training
    series, which scales the error so series of different magnitudes are
    comparable. Lower is better; 1.0 means "as good as a naive forecast".
    """
    naive_err = np.mean(np.diff(train_actual) ** 2)
    if naive_err == 0:
        return np.nan
    return np.sqrt(np.mean((actual - forecast) ** 2) / naive_err)


def evaluate_series(
    test: pd.DataFrame, train: pd.DataFrame, actual_col: str = "units",
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    """Per-series RMSSE over the held-out horizon."""
    rows = []
    for series_id, grp in test.groupby("item_store"):
        train_grp = train[train["item_store"] == series_id].sort_values("date")
        if len(train_grp) < 2:
            continue
        rows.append({
            "item_store": series_id,
            "rmsse": rmsse(
                grp[actual_col].to_numpy(),
                grp[forecast_col].to_numpy(),
                train_grp[actual_col].to_numpy(),
            ),
        })
    out = pd.DataFrame(rows)
    logger.info("evaluated %d series, mean RMSSE %.3f", len(out), out["rmsse"].mean())
    return out


def financial_translation(
    forecast: pd.DataFrame,
    prices: pd.DataFrame,
    understock_penalty: float = 1.0,
    overstock_penalty: float = 0.25,
    actual_col: str = "units",
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    """
    Translate units into money: expected revenue and asymmetric forecast-error
    cost.

    The two penalties encode a real planning trade-off. Under-forecasting
    causes stockouts -- you lose the full margin on the sale you could not
    make (`understock_penalty`, as a fraction of price). Over-forecasting ties
    up working capital and risks waste, but usually costs less per unit
    (`overstock_penalty`). The asymmetry is the point: a planner does not
    treat the two errors equally, and neither should the system.

    Returns a per-row frame with revenue and error-cost, ready to aggregate
    to any level for an executive brief.
    """
    # The forecast frame may already carry sell_price from feature building.
    if "sell_price" in forecast.columns:
        df = forecast.copy()
    else:
        df = forecast.merge(
            prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
            on=["store_id", "item_id", "wm_yr_wk"], how="left",
        )
    df["sell_price"] = df.groupby("item_store")["sell_price"].ffill().fillna(0)
    df["expected_revenue"] = df[forecast_col] * df["sell_price"]

    error = df[forecast_col] - df[actual_col]
    over = error.clip(lower=0)     # forecast too high
    under = (-error).clip(lower=0)  # forecast too low
    df["error_cost"] = (
        over * df["sell_price"] * overstock_penalty
        + under * df["sell_price"] * understock_penalty
    )
    logger.info(
        "financial translation: revenue %.0f, error cost %.0f",
        df["expected_revenue"].sum(), df["error_cost"].sum(),
    )
    return df


def executive_summary(financial: pd.DataFrame) -> dict:
    """Roll the financial layer into the handful of numbers an executive reads."""
    total_units = financial["forecast"].sum()
    total_rev = financial["expected_revenue"].sum()
    total_cost = financial["error_cost"].sum()
    return {
        "forecast_horizon_days": financial["date"].nunique(),
        "expected_units": round(total_units),
        "expected_revenue": round(total_rev, 2),
        "forecast_error_cost": round(total_cost, 2),
        "error_cost_pct_of_revenue": round(100 * total_cost / total_rev, 2) if total_rev else None,
    }
