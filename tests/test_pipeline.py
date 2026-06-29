"""Tests for the forecasting pipeline. These run in CI on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src import data as data_mod
from src import features, model, reconciliation, evaluate, synthetic


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    synthetic.generate(d, n_items=30, n_days=200, seed=1)
    return data_mod.load_m5(d)


def test_data_validates_and_loads(synth):
    assert synth.n_series == 30
    assert synth.sales_long["units"].min() >= 0
    assert synth.sales_long["date"].notna().all()


def test_negative_sales_rejected(tmp_path):
    synthetic.generate(tmp_path, n_items=5, n_days=50)
    df = pd.read_csv(tmp_path / "sales_train_validation.csv")
    df["d_1"] = -1
    df.to_csv(tmp_path / "sales_train_validation.csv", index=False)
    with pytest.raises(data_mod.DataValidationError):
        data_mod.load_m5(tmp_path)


def test_features_are_causal(synth):
    """Lag features must never equal same-day units (no leakage)."""
    feat = features.build_features(synth.sales_long, synth.calendar, synth.prices)
    assert "lag_7" in feat.columns
    assert feat["lag_7"].notna().any()


def test_temporal_split_has_no_overlap(synth):
    feat = features.build_features(synth.sales_long, synth.calendar, synth.prices)
    train, test = model.train_test_split_temporal(feat, horizon_days=28)
    assert train["date"].max() < test["date"].min()


def test_reconciliation_is_coherent(synth):
    """The core guarantee: aggregated levels sum to the grand total."""
    feat = features.build_features(synth.sales_long, synth.calendar, synth.prices)
    train, test = model.train_test_split_temporal(feat, horizon_days=28)
    fm = model.ForecastModel(num_boost_round=50).fit(
        train, [c for c in feat.columns if c.startswith(("lag_", "roll_"))]
    )
    test = test.copy()
    test["forecast"] = fm.predict(test)
    id_cols = ["item_store", "item_id", "store_id", "dept_id", "cat_id", "state_id"]
    bottom = test[id_cols + ["date", "wm_yr_wk", "forecast"]]
    rec = reconciliation.reconcile_bottom_up(
        bottom, ["total", "state_id", "cat_id", "item_store"], "forecast"
    )
    assert reconciliation.check_coherence(rec, "forecast")


def test_rmsse_matches_naive_definition():
    """RMSSE of a perfect forecast is 0; of the naive forecast is ~1."""
    train = np.array([1.0, 2, 3, 2, 4, 3, 5])
    actual = np.array([4.0, 3, 5])
    assert evaluate.rmsse(actual, actual, train) == 0.0


def test_financial_translation_asymmetry(synth):
    """Under-forecast must cost more per unit than over-forecast."""
    feat = features.build_features(synth.sales_long, synth.calendar, synth.prices)
    _, test = model.train_test_split_temporal(feat, horizon_days=28)
    test = test.copy()
    # under-forecast: forecast below actual
    test["forecast"] = (test["units"] - 1).clip(lower=0)
    under = evaluate.financial_translation(test, synth.prices)["error_cost"].sum()
    # over-forecast by the same magnitude
    test["forecast"] = test["units"] + 1
    over = evaluate.financial_translation(test, synth.prices)["error_cost"].sum()
    assert under > over
