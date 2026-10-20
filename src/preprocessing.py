"""Cleaning and validation of the hourly workload time series.

Input  : a raw DataFrame with columns ``timestamp`` and ``requests``.
Output : a cleaned DataFrame with one row for EVERY hour in the range
         (no gaps), columns ``timestamp``, ``requests``, ``imputed``.

Why this is needed: ARIMA assumes equally spaced observations. If an hour is
missing in the download, the series must be repaired before modelling.
"""

from __future__ import annotations

import pandas as pd

# A week of hourly data is the smallest series we accept at all.
MIN_OBSERVATIONS = 168
# If more than 20% of the hours are missing, the data is too poor to trust.
MAX_MISSING_FRACTION = 0.20


class PreprocessingError(Exception):
    """Raised when the data cannot be turned into a usable time series."""


def clean_workload(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw hourly pageview data.

    Steps
    -----
    1. Validate columns and convert types (bad values become NaN).
    2. Sort by time and drop duplicate timestamps.
    3. Re-index to a complete hourly range so missing hours become visible.
    4. Treat non-positive counts as missing (a zero for the whole English
       Wikipedia in one hour means a measurement gap, not real traffic).
    5. Fill the gaps by time-based linear interpolation.
    6. Check that enough data is left.
    """
    if raw is None or raw.empty:
        raise PreprocessingError("No data received - the dataset is empty.")

    missing_cols = {"timestamp", "requests"} - set(raw.columns)
    if missing_cols:
        raise PreprocessingError(f"Missing required column(s): {sorted(missing_cols)}")

    df = raw[["timestamp", "requests"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["requests"] = pd.to_numeric(df["requests"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise PreprocessingError("No valid timestamps found in the data.")

    # Remove timezone information (all Wikimedia timestamps are UTC) so that
    # the index is a plain datetime index.
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="last")
        .set_index("timestamp")
    )

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name="timestamp")
    df = df.reindex(full_index)

    df.loc[df["requests"] <= 0, "requests"] = float("nan")
    missing_mask = df["requests"].isna()

    missing_fraction = float(missing_mask.mean())
    if missing_fraction > MAX_MISSING_FRACTION:
        raise PreprocessingError(
            f"{missing_fraction:.0%} of the hourly values are missing "
            f"(limit is {MAX_MISSING_FRACTION:.0%}). Try refreshing the data."
        )

    df["requests"] = df["requests"].interpolate(method="time", limit_direction="both")
    df["imputed"] = missing_mask

    if len(df) < MIN_OBSERVATIONS:
        raise PreprocessingError(
            f"Only {len(df)} hourly observations available; at least "
            f"{MIN_OBSERVATIONS} (one week) are required."
        )

    return df.reset_index()[["timestamp", "requests", "imputed"]]


def to_series(clean_df: pd.DataFrame) -> pd.Series:
    """Convert the cleaned DataFrame to a pandas Series with an hourly index.

    statsmodels needs an index with a known frequency to forecast correctly.
    """
    series = pd.Series(
        clean_df["requests"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(clean_df["timestamp"], name="timestamp"),
        name="requests",
    )
    return series.asfreq("h")
