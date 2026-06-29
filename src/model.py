"""
Forecasting model: LightGBM over engineered features.

The published M5 post-mortems converge on gradient-boosted trees as the
workhorse for this hierarchical, intermittent-demand problem -- they capture
the categorical interactions (product x store) that traditional univariate
methods (ARIMA, ETS) miss, and they handle the many-zeros intermittency that
is typical of store-level retail sales.

The model is wrapped behind a small interface so the rest of the system
(reconciliation, evaluation, the financial layer) does not depend on the
modeling library. Swapping LightGBM for another learner touches only this
file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "objective": "tweedie",      # tweedie handles intermittent (many-zero) demand
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


@dataclass
class ForecastModel:
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 500
    _model: object = None
    _features: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame, features: list[str], target: str = "units") -> "ForecastModel":
        import lightgbm as lgb

        self._features = features
        dtrain = lgb.Dataset(train[features], label=train[target])
        self._model = lgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round
        )
        logger.info("model trained on %d rows, %d features", len(train), len(features))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        preds = self._model.predict(df[self._features])
        return np.clip(preds, 0, None)  # unit sales cannot be negative

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return (
            pd.DataFrame(
                {"feature": self._features,
                 "importance": self._model.feature_importance(importance_type="gain")}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def train_test_split_temporal(
    df: pd.DataFrame, horizon_days: int = 28
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time, not at random. The last `horizon_days` are held out as the
    test set, mirroring the real forecasting task: predict the next 28 days
    given everything before.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=horizon_days)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    logger.info(
        "temporal split at %s: %d train, %d test rows",
        cutoff.date(), len(train), len(test)
    )
    return train, test
