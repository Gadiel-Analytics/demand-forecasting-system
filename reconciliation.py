"""
Hierarchical reconciliation.

This is what separates a forecasting *system* from a forecasting notebook.

In retail, forecasts are consumed at many levels at once: a store manager
plans at the store level, a category buyer at the category level, finance at
the total level. If the store-level forecasts do not sum to the category
forecast, and the category forecasts do not sum to the total, then different
parts of the business plan against numbers that contradict each other. The
forecasts are *incoherent*, and the decisions made from them are misaligned.

Reconciliation enforces coherence: it adjusts the base forecasts so that
every level sums correctly to the level above. This module implements
bottom-up reconciliation (sum the bottom level to produce all aggregates),
which is simple, robust, and the natural baseline. The structure also
documents where MinT (minimum-trace) optimal reconciliation would slot in
for a production upgrade.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Mapping from a hierarchy level name to the id columns that define it.
LEVEL_KEYS = {
    "total": [],
    "state_id": ["state_id"],
    "store_id": ["store_id"],
    "cat_id": ["cat_id"],
    "dept_id": ["dept_id"],
    "state_cat": ["state_id", "cat_id"],
    "store_dept": ["store_id", "dept_id"],
    "item_id": ["item_id"],
    "item_store": ["item_store"],
}


def aggregate_to_level(bottom: pd.DataFrame, level: str, value_col: str) -> pd.DataFrame:
    """
    Sum bottom-level (item_store) forecasts up to a named hierarchy level.

    Because every aggregate is computed by summing the same bottom-level
    numbers, the result is coherent by construction: store totals sum to
    category totals, which sum to the grand total.
    """
    keys = LEVEL_KEYS[level]
    if not keys:  # total
        out = bottom.groupby("date", as_index=False)[value_col].sum()
        out["level"] = "total"
        return out
    out = bottom.groupby(keys + ["date"], as_index=False)[value_col].sum()
    out["level"] = level
    return out


def reconcile_bottom_up(
    bottom: pd.DataFrame, levels: list[str], value_col: str = "forecast"
) -> dict[str, pd.DataFrame]:
    """
    Produce coherent forecasts at every requested level from the bottom level.

    Parameters
    ----------
    bottom
        Bottom-level (item_store x date) forecasts. Must carry the id columns
        needed to roll up: item_store, item_id, store_id, dept_id, cat_id,
        state_id.
    levels
        Hierarchy level names to produce (see LEVEL_KEYS).
    value_col
        Name of the forecast column to aggregate.

    Returns
    -------
    dict
        Level name -> coherent forecast frame.
    """
    out: dict[str, pd.DataFrame] = {}
    for level in levels:
        out[level] = aggregate_to_level(bottom, level, value_col)
        logger.info("reconciled level '%s': %d rows", level, len(out[level]))
    return out


def check_coherence(
    reconciled: dict[str, pd.DataFrame], value_col: str = "forecast", tol: float = 1e-6
) -> bool:
    """
    Verify that the reconciled forecasts are coherent: every level's total
    equals the grand total on each date. Returns True if coherent.

    This is the assertion a production pipeline runs before publishing -- if
    coherence fails, the forecasts are not safe to plan against.
    """
    if "total" not in reconciled:
        raise ValueError("coherence check requires the 'total' level")
    grand = reconciled["total"].set_index("date")[value_col]

    for level, frame in reconciled.items():
        if level == "total":
            continue
        level_total = frame.groupby("date")[value_col].sum()
        diff = (level_total - grand).abs()
        if (diff > tol).any():
            bad = diff[diff > tol]
            logger.error("level '%s' incoherent on %d dates", level, len(bad))
            return False
    logger.info("coherence check passed across %d levels", len(reconciled))
    return True
