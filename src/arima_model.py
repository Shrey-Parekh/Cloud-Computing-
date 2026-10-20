"""ARIMA training, test-period prediction and future forecasting.

ARIMA(p, d, q):
    p = number of autoregressive (AR) lags     - past values used as inputs
    d = number of differencing steps           - makes the series stationary
    q = number of moving-average (MA) lags     - past forecast errors used

Two ways to predict the test period are offered:

* ``one_step``  - rolling one-step-ahead: every hour is predicted from all the
  real observations before it. The model parameters stay fixed (fitted on the
  training data only), but the newest real values are fed in as they "arrive".
  This is how a cloud provisioner works in practice: forecast the next
  interval, then observe it.
* ``multi_step`` - the model forecasts the whole test period in one go without
  seeing any test values. This is much harder, so errors are larger.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # noqa: F401  (documented)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

MODE_ONE_STEP = "one_step"
MODE_MULTI_STEP = "multi_step"
SEASONAL = (1, 0, 1, 24)

MAX_P, MAX_D, MAX_Q = 10, 2, 10
MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 24


class ArimaError(Exception):
    """Raised for invalid ARIMA settings or when a model cannot be fitted."""


# --------------------------------------------------------------------------- #
# Validation and splitting
# --------------------------------------------------------------------------- #
def validate_order(order: Iterable[int], n_obs: int) -> tuple:
    """Check (p, d, q) and return it as a tuple of ints."""
    try:
        values = tuple(order)
        if len(values) != 3 or any(float(v) != int(v) for v in values):
            raise ValueError
        p, d, q = (int(v) for v in values)
    except (TypeError, ValueError):
        raise ArimaError("ARIMA order must be three whole numbers (p, d, q).") from None

    if not (0 <= p <= MAX_P and 0 <= d <= MAX_D and 0 <= q <= MAX_Q):
        raise ArimaError(
            f"Order out of range: p and q must be 0-{MAX_P}, d must be 0-{MAX_D}."
        )
    if n_obs < 3 * (p + d + q) + 30:
        raise ArimaError(
            f"Not enough data ({n_obs} rows) for ARIMA({p},{d},{q}). "
            "Use smaller p/d/q or more data."
        )
    return p, d, q


def split_train_test(series: pd.Series, train_ratio: float = 0.8):
    """Chronological split (never shuffle a time series!)."""
    if not 0.5 <= train_ratio <= 0.95:
        raise ArimaError("Train ratio must be between 0.5 and 0.95.")
    n_train = int(len(series) * train_ratio)
    n_test = len(series) - n_train
    if n_train < MIN_TRAIN_ROWS or n_test < MIN_TEST_ROWS:
        raise ArimaError(
            f"Insufficient data: need at least {MIN_TRAIN_ROWS} training and "
            f"{MIN_TEST_ROWS} test rows, got {n_train} and {n_test}."
        )
    return series.iloc[:n_train], series.iloc[n_train:]


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_arima(train: pd.Series, order, seasonal_order=SEASONAL):
    """Fit ARIMA on ``train`` and return the statsmodels results object."""
    order = validate_order(order, len(train))
    try:
        with warnings.catch_warnings():
            # Convergence / start-parameter warnings are common for hourly data;
            # the fit is still usable, so we do not flood the dashboard.
            warnings.simplefilter("ignore")
            return ARIMA(train, order=order, seasonal_order=seasonal_order).fit()
    except ArimaError:
        raise
    except Exception as exc:  # statsmodels raises many different error types
        raise ArimaError(
            f"ARIMA{order} could not be fitted ({type(exc).__name__}: {exc}). "
            "Try a different (p, d, q)."
        ) from exc


def predict_test_period(train: pd.Series, test: pd.Series, order,
                        mode: str = MODE_ONE_STEP):
    """Train on ``train`` and predict the test period.

    Returns (predicted_series_aligned_with_test, aic_of_training_fit).
    """
    if mode not in (MODE_ONE_STEP, MODE_MULTI_STEP):
        raise ArimaError(f"Unknown forecast mode: {mode!r}")

    fitted = fit_arima(train, order, seasonal_order=SEASONAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if mode == MODE_MULTI_STEP:
                values = fitted.get_forecast(steps=len(test)).predicted_mean
            else:
                # Add the test observations WITHOUT re-estimating parameters,
                # then read the one-step-ahead predictions for that period.
                extended = fitted.append(test, refit=False)
                values = extended.get_prediction(
                    start=len(train), end=len(train) + len(test) - 1
                ).predicted_mean
    except Exception as exc:
        raise ArimaError(f"Prediction failed ({type(exc).__name__}: {exc}).") from exc

    predicted = pd.Series(np.asarray(values, dtype=float), index=test.index, name="predicted")
    predicted = predicted.clip(lower=0)  # traffic cannot be negative
    if predicted.isna().any():
        raise ArimaError("The model produced NaN predictions; try another (p, d, q).")
    return predicted, float(fitted.aic)


def forecast_future(series: pd.Series, order, steps: int, alpha: float = 0.05):
    """Refit a daily seasonal ARIMA on ALL data and forecast the next ``steps`` hours.

    The (1, 0, 1, 24) seasonal component captures recurring hourly patterns
    across days. Returns predicted, lower, upper (95% interval).
    """
    if not isinstance(steps, (int, np.integer)) or steps < 1 or steps > 168:
        raise ArimaError("Forecast horizon must be a whole number between 1 and 168 hours.")

    fitted = fit_arima(series, order, seasonal_order=SEASONAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fitted.get_forecast(steps=int(steps))
            mean = np.asarray(result.predicted_mean, dtype=float)
            interval = np.asarray(result.conf_int(alpha=alpha), dtype=float)
    except Exception as exc:
        raise ArimaError(f"Forecast failed ({type(exc).__name__}: {exc}).") from exc

    future_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(hours=1), periods=int(steps),
        freq="h", name="timestamp",
    )
    out = pd.DataFrame(
        {"predicted": mean, "lower": interval[:, 0], "upper": interval[:, 1]},
        index=future_index,
    ).clip(lower=0)
    if out["predicted"].isna().any():
        raise ArimaError("The model produced NaN forecasts; try another (p, d, q).")
    return out


# --------------------------------------------------------------------------- #
# Optional helper: pick an order automatically
# --------------------------------------------------------------------------- #
def choose_d(train: pd.Series) -> int:
    """Pick d with the Augmented Dickey-Fuller test (p < 0.05 => stationary)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels announces a future return-type change
            p_value = adfuller(train.to_numpy(), autolag="AIC")[1]
    except Exception:
        return 1
    return 0 if p_value < 0.05 else 1


def suggest_order(train: pd.Series, max_p: int = 3, max_q: int = 2):
    """Choose d by ADF test, then (p, q) with the lowest AIC on the training data.

    Returns (best_order, table_of_all_candidates_sorted_by_AIC).
    """
    d = choose_d(train)
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                aic = float(fit_arima(train, (p, d, q)).aic)
            except ArimaError:
                continue
            rows.append({"p": p, "d": d, "q": q, "AIC": aic})
    if not rows:
        raise ArimaError("No candidate ARIMA order could be fitted.")
    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    best = table.iloc[0]
    return (int(best["p"]), int(best["d"]), int(best["q"])), table
