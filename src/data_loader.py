"""Download real hourly English Wikipedia traffic from the Wikimedia API.

Data source (official, no API key needed):

    https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/
        {project}/{access}/{agent}/{granularity}/{start}/{end}

We use project=en.wikipedia, access=all-access, agent=user (human traffic,
bots/spiders excluded), granularity=hourly. Timestamps look like
``2026090100`` (YYYYMMDDHH, UTC). Each item in the JSON reply holds the number
of page views ("views") for that hour.

If Wikimedia ever changes this endpoint, only API_BASE below needs updating.

Run ``python -m src.data_loader`` to download the data from the command line.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .preprocessing import MIN_OBSERVATIONS, PreprocessingError, clean_workload

logger = logging.getLogger(__name__)

API_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate"
PROJECT = "en.wikipedia"
ACCESS = "all-access"
AGENT = "user"
GRANULARITY = "hourly"

DEFAULT_DAYS = 30            # 30 days * 24 h = 720 hourly rows
DATA_LAG_HOURS = 6           # the newest hours are not published immediately
CACHE_MAX_AGE_HOURS = 6      # do not download again if the CSV is newer than this
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "workload.csv"

# Wikimedia asks API clients to send a descriptive User-Agent. You can put your
# own contact in the WIKI_USER_AGENT environment variable.
USER_AGENT = os.environ.get(
    "WIKI_USER_AGENT",
    "CloudWorkloadPrediction/1.0 (college cloud-computing project; "
    "https://wikimedia.org/api/rest_v1/)",
)


class DataLoadError(Exception):
    """Raised when the workload data cannot be downloaded or read."""


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def build_url(start: datetime, end: datetime) -> str:
    """Build the API URL for an inclusive hourly range."""
    return (
        f"{API_BASE}/{PROJECT}/{ACCESS}/{AGENT}/{GRANULARITY}/"
        f"{start:%Y%m%d%H}/{end:%Y%m%d%H}"
    )


def parse_response(payload: dict) -> pd.DataFrame:
    """Turn the API's JSON into a DataFrame(timestamp, requests)."""
    items = payload.get("items") if isinstance(payload, dict) else None
    if not items:
        raise DataLoadError("Wikimedia API reply contained no data ('items' is empty).")
    try:
        df = pd.DataFrame(
            {
                "timestamp": [str(item["timestamp"]) for item in items],
                "requests": [item["views"] for item in items],
            }
        )
    except (KeyError, TypeError) as exc:
        raise DataLoadError(f"Unexpected API reply format (missing field {exc}).") from exc

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H", errors="coerce")
    return df


def download_hourly_pageviews(days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """Download ``days`` days of hourly pageviews. Retries on temporary errors."""
    if not isinstance(days, int) or not 8 <= days <= 90:
        raise DataLoadError("Number of days must be a whole number between 8 and 90.")

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = now - timedelta(hours=DATA_LAG_HOURS)
    start = end - timedelta(hours=days * 24 - 1)
    url = build_url(start, end)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    last_problem = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_problem = f"network problem ({type(exc).__name__})"
        else:
            if response.status_code == 200:
                try:
                    return parse_response(response.json())
                except ValueError as exc:  # invalid JSON
                    raise DataLoadError("Wikimedia API returned invalid JSON.") from exc
            if response.status_code in (429, 500, 502, 503, 504):
                last_problem = f"temporary server error (HTTP {response.status_code})"
            else:  # 4xx: retrying will not help
                raise DataLoadError(
                    f"Wikimedia API returned HTTP {response.status_code} for {url}"
                )
        logger.warning("Download attempt %d/%d failed: %s", attempt, MAX_RETRIES, last_problem)
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)

    raise DataLoadError(
        f"Could not download data after {MAX_RETRIES} attempts: {last_problem}."
    )


# --------------------------------------------------------------------------- #
# CSV cache
# --------------------------------------------------------------------------- #
def save_clean_csv(df: pd.DataFrame, path: Path = DEFAULT_CSV_PATH) -> None:
    """Save cleaned data (write to a temp file first so a crash cannot corrupt it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_clean_csv(path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Read a previously saved cleaned CSV and validate it."""
    path = Path(path)
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataLoadError(f"Could not read {path.name}: {exc}") from exc
    if not {"timestamp", "requests"} <= set(df.columns) or len(df) < MIN_OBSERVATIONS:
        raise DataLoadError(f"{path.name} is incomplete or has an unexpected format.")
    if "imputed" not in df.columns:
        df["imputed"] = False
    df["imputed"] = df["imputed"].astype(bool)
    return df


def _cache_age_hours(path: Path) -> float:
    return (time.time() - Path(path).stat().st_mtime) / 3600.0


# --------------------------------------------------------------------------- #
# Main entry point used by the dashboard
# --------------------------------------------------------------------------- #
def get_workload(days: int = DEFAULT_DAYS, csv_path: Path = DEFAULT_CSV_PATH,
                 force_refresh: bool = False):
    """Return (cleaned_dataframe, info_dict).

    Order of attempts:
      1. Use data/workload.csv if it exists and is recent (no download).
      2. Otherwise download from Wikimedia, clean, and save to data/workload.csv.
      3. If the download fails but an older CSV exists, use it and warn.
      4. Otherwise raise DataLoadError / PreprocessingError.
    """
    csv_path = Path(csv_path)

    if csv_path.exists() and not force_refresh and _cache_age_hours(csv_path) < CACHE_MAX_AGE_HOURS:
        try:
            df = load_clean_csv(csv_path)
            return df, {"source": "cache", "message": "Loaded from the local CSV (recent)."}
        except DataLoadError:
            logger.warning("Local CSV unusable; downloading fresh data.")

    try:
        df = clean_workload(download_hourly_pageviews(days))
        save_clean_csv(df, csv_path)
        return df, {"source": "download", "message": "Downloaded fresh data from Wikimedia."}
    except (DataLoadError, PreprocessingError) as exc:
        if csv_path.exists():
            try:
                df = load_clean_csv(csv_path)
            except DataLoadError:
                raise exc
            return df, {
                "source": "stale-cache",
                "message": f"Download failed ({exc}). Showing the older local CSV instead.",
            }
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data, info = get_workload(force_refresh=True)
    print(info["message"])
    print(f"Rows: {len(data)} | from {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Imputed (filled) hours: {int(data['imputed'].sum())}")
    print(f"Saved to: {DEFAULT_CSV_PATH}")
