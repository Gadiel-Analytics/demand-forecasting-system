"""
Synthetic M5-like data generator.

The real M5 files are ~450 MB and require a Kaggle account, which makes a repo
that depends on them hard to run and impossible to test in CI. This generator
produces a small, structurally faithful stand-in: the same column schema, the
same hierarchy (states > stores, categories > departments > items), calendar
events, SNAP flags, and intermittent (many-zero) demand with weekly
seasonality and price effects.

It lets anyone clone the repo and run the full pipeline end to end in seconds,
and it gives CI a deterministic fixture. Production runs point at the real M5
files instead (see DEPLOYMENT.md). This mirrors the resilience principle from
the flagship: the system should run for any visitor, not only for its author.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate(out_dir: str | Path, n_items: int = 50, n_days: int = 400, seed: int = 42) -> None:
    """Write synthetic sales_train_validation.csv, calendar.csv, sell_prices.csv."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = ["CA", "TX", "WI"]
    stores = {s: [f"{s}_{i+1}" for i in range(2)] for s in states}
    all_stores = [st for sts in stores.values() for st in sts]
    cats = {"FOODS": ["FOODS_1", "FOODS_2"], "HOBBIES": ["HOBBIES_1"], "HOUSEHOLD": ["HOUSEHOLD_1"]}

    # ---- calendar ----
    start = pd.Timestamp("2014-01-01")
    dates = pd.date_range(start, periods=n_days, freq="D")
    cal = pd.DataFrame({
        "date": dates.astype(str),
        "d": [f"d_{i+1}" for i in range(n_days)],
        "wm_yr_wk": [11400 + (i // 7) for i in range(n_days)],
    })
    cal["event_name_1"] = np.where(rng.random(n_days) < 0.05, "Event", None)
    for s in states:
        cal[f"snap_{s}"] = (rng.random(n_days) < 0.33).astype(int)

    # ---- items and sales ----
    rows, price_rows = [], []
    for k in range(n_items):
        cat = rng.choice(list(cats.keys()))
        dept = rng.choice(cats[cat])
        item = f"{dept}_{k+1:03d}"
        store = rng.choice(all_stores)
        state = store.split("_")[0]

        base = rng.uniform(0.5, 8.0)          # base daily demand
        dow_amp = rng.uniform(0.2, 0.6)       # weekly seasonality strength
        price = round(rng.uniform(1.5, 12.0), 2)

        demand = []
        for i, day in enumerate(dates):
            season = 1 + dow_amp * np.sin(2 * np.pi * day.dayofweek / 7)
            lam = max(base * season, 0.05)
            units = rng.poisson(lam)           # intermittent count demand
            demand.append(units)

        row = {
            "id": f"{item}_{store}_validation",
            "item_id": item, "dept_id": dept, "cat_id": cat,
            "store_id": store, "state_id": state,
        }
        row.update({f"d_{i+1}": demand[i] for i in range(n_days)})
        rows.append(row)

        for wk in cal["wm_yr_wk"].unique():
            price_rows.append({"store_id": store, "item_id": item,
                               "wm_yr_wk": int(wk), "sell_price": price})

    pd.DataFrame(rows).to_csv(out_dir / "sales_train_validation.csv", index=False)
    cal.to_csv(out_dir / "calendar.csv", index=False)
    pd.DataFrame(price_rows).to_csv(out_dir / "sell_prices.csv", index=False)
    print(f"wrote synthetic M5 data to {out_dir} ({n_items} items, {n_days} days)")


if __name__ == "__main__":
    generate("data")
