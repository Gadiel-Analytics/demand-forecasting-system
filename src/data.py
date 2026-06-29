"""
Data loading and validation for the M5 hierarchical retail dataset.

The M5 dataset (Walmart, 2011-2016) is the industry-standard benchmark for
hierarchical retail demand forecasting: 30,490 bottom-level series across
3,049 products, 10 stores, and 3 US states, with calendar events, SNAP
indicators, and sell prices.

This module loads the three source files, validates their integrity against
an explicit schema, and reshapes the wide sales matrix into a long,
model-ready frame. Validation is deliberate: a forecasting system that
silently ingests malformed data produces confident, wrong forecasts -- the
most expensive failure mode in demand planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# The three canonical M5 files. See DEPLOYMENT.md for how to obtain them.
SALES_FILE = "sales_train_validation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

# Hierarchy levels, from most aggregate to most granular. The system produces
# coherent forecasts across all of these (see reconciliation.py).
HIERARCHY_LEVELS = [
    "total",          # 1 series:  all sales
    "state_id",       # 3 series:  CA, TX, WI
    "store_id",       # 10 series
    "cat_id",         # 3 series:  Hobbies, Foods, Household
    "dept_id",        # 7 series
    "state_cat",      # state x category
    "store_dept",     # store x department
    "item_id",        # 3,049 series
    "item_store",     # 30,490 series: the bottom level
]


@dataclass
class M5Data:
    """Validated, model-ready M5 data."""

    sales_long: pd.DataFrame   # one row per (item_store, day)
    calendar: pd.DataFrame     # one row per day, with events and SNAP flags
    prices: pd.DataFrame       # one row per (store, item, week)

    @property
    def n_series(self) -> int:
        return self.sales_long["item_store"].nunique()

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.sales_long["date"].min(), self.sales_long["date"].max()


class DataValidationError(Exception):
    """Raised when source data fails an integrity check."""


def _validate_sales(df: pd.DataFrame) -> None:
    required = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"sales file missing id columns: {missing}")
    day_cols = [c for c in df.columns if c.startswith("d_")]
    if not day_cols:
        raise DataValidationError("sales file has no day (d_*) columns")
    if df[day_cols].isnull().any().any():
        raise DataValidationError("sales file contains null sales values")
    if (df[day_cols] < 0).any().any():
        raise DataValidationError("sales file contains negative unit sales")
    logger.info("sales validated: %d series, %d days", len(df), len(day_cols))


def _validate_calendar(df: pd.DataFrame) -> None:
    required = {"date", "d", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"calendar file missing columns: {missing}")
    if df["d"].duplicated().any():
        raise DataValidationError("calendar has duplicate day identifiers")
    logger.info("calendar validated: %d days", len(df))


def _validate_prices(df: pd.DataFrame) -> None:
    required = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"prices file missing columns: {missing}")
    if (df["sell_price"] <= 0).any():
        raise DataValidationError("prices file contains non-positive prices")
    logger.info("prices validated: %d price points", len(df))


def load_m5(data_dir: str | Path, sample_frac: float | None = None) -> M5Data:
    """
    Load and validate the M5 dataset into a model-ready form.

    Parameters
    ----------
    data_dir
        Directory containing the three M5 CSV files.
    sample_frac
        If set (0 < frac <= 1), randomly sample this fraction of bottom-level
        series. Useful for fast local iteration and CI; production runs use
        the full set.

    Returns
    -------
    M5Data
        Validated long-format sales, calendar, and prices.
    """
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / SALES_FILE)
    calendar = pd.read_csv(data_dir / CALENDAR_FILE)
    prices = pd.read_csv(data_dir / PRICES_FILE)

    _validate_sales(sales)
    _validate_calendar(calendar)
    _validate_prices(prices)

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1]")
        sales = sales.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        logger.info("sampled %.1f%% of series -> %d", sample_frac * 100, len(sales))

    sales["item_store"] = sales["item_id"] + "__" + sales["store_id"]

    id_cols = ["item_store", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    sales_long = sales.melt(
        id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="units"
    )
    sales_long = sales_long.merge(
        calendar[["d", "date", "wm_yr_wk"]], on="d", how="left"
    )
    sales_long["date"] = pd.to_datetime(sales_long["date"])

    logger.info(
        "loaded M5: %d series, %s to %s",
        sales_long["item_store"].nunique(),
        sales_long["date"].min().date(),
        sales_long["date"].max().date(),
    )
    return M5Data(sales_long=sales_long, calendar=calendar, prices=prices)
