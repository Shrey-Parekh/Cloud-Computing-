"""Forecast accuracy metrics: MAE, RMSE, MAPE and project accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd


class EvaluationError(Exception):
    """Raised when metrics cannot be computed."""


def mae(actual, predicted) -> float:
    """Mean Absolute Error = mean(|actual - predicted|). Same unit as the data."""
    a, p = _to_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    """Root Mean Squared Error = sqrt(mean((actual - predicted)^2)).

    Squaring penalises large errors more than MAE does.
    """
    a, p = _to_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted) -> float:
    """Mean Absolute Percentage Error (in %) = mean(|a - p| / |a|) * 100.

    Hours where the actual value is exactly 0 are skipped (division by zero).
    """
    a, p = _to_arrays(actual, predicted)
    nonzero = a != 0
    if not nonzero.any():
        raise EvaluationError("MAPE is undefined: all actual values are zero.")
    return float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100.0)


def evaluate(actual, predicted) -> dict:
    """Return all metrics in one dictionary.

    ``accuracy`` is the project's simple accuracy figure: 100 - MAPE.
    It is NOT clipped, so a very bad model can show a value below 0.
    """
    m = mape(actual, predicted)
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": m,
        "Accuracy": 100.0 - m,
    }


def _to_arrays(actual, predicted):
    a = np.asarray(pd.Series(actual).to_numpy(), dtype=float)
    p = np.asarray(pd.Series(predicted).to_numpy(), dtype=float)
    if a.shape != p.shape:
        raise EvaluationError(
            f"Length mismatch: {a.shape[0]} actual vs {p.shape[0]} predicted values."
        )
    if a.size == 0:
        raise EvaluationError("Cannot evaluate empty series.")
    if np.isnan(a).any() or np.isnan(p).any():
        raise EvaluationError("Metrics cannot be computed: NaN values present.")
    return a, p
