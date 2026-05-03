"""
RideWise — Data Preprocessing
==============================
Cleans all raw datasets and saves them to data/processed/.

Usage:
    python -m src.data.preprocessing
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)


# ── Individual cleaning functions ────────────────────────────
def clean_riders(riders: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the riders dataset.
    - Round age to integer
    - Convert churn_prob to binary label (threshold = 0.5)
    - Encode referred_by as binary was_referred flag
    """
    df = riders.copy()
    df["age"]          = df["age"].round().astype("Int64")
    df["churned"]      = (df["churn_prob"] > 0.5).astype(int)
    df                 = df.drop(columns=["churn_prob"])
    df["was_referred"] = df["referred_by"].notna().astype(int)
    df                 = df.drop(columns=["referred_by"])
    return df


def clean_trips(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the trips dataset.
    - Parse timezone-aware timestamps
    - Engineer trip_duration, total_revenue, hour_of_day, day_of_week
    - Remove non-positive durations
    - Fill missing tips with 0
    """
    df = trips.copy()

    for col in ["pickup_time", "dropoff_time"]:
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(None)

    df = df.dropna(subset=["pickup_time", "dropoff_time"])

    df["trip_duration"] = (
        (df["dropoff_time"] - df["pickup_time"]).dt.total_seconds() / 60
    )
    df["total_revenue"] = df["fare"] * df["surge_multiplier"] + df["tip"].fillna(0)
    df["hour_of_day"]   = df["pickup_time"].dt.hour
    df["day_of_week"]   = df["pickup_time"].dt.day_name()

    df          = df[df["trip_duration"] > 0]
    df["tip"]   = df["tip"].fillna(0)

    return df


def clean_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the drivers dataset.
    - Impute missing rating and acceptance_rate with median
    """
    df = drivers.copy()
    df["rating"]          = df["rating"].fillna(df["rating"].median())
    df["acceptance_rate"] = df["acceptance_rate"].fillna(df["acceptance_rate"].median())
    return df


def clean_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the sessions dataset.
    - Parse timezone-aware session_time timestamps
    """
    df = sessions.copy()
    df["session_time"] = pd.to_datetime(df["session_time"], utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["session_time"])
    return df


def check_referential_integrity(
    riders_clean: pd.DataFrame,
    drivers_clean: pd.DataFrame,
    trips_clean: pd.DataFrame,
    sessions_clean: pd.DataFrame,
) -> tuple:
    """
    Remove trips/sessions whose user_id or driver_id
    does not exist in the riders/drivers tables.
    """
    valid_riders  = set(riders_clean["user_id"])
    valid_drivers = set(drivers_clean["driver_id"])

    before_trips    = len(trips_clean)
    before_sessions = len(sessions_clean)

    trips_clean    = trips_clean[trips_clean["user_id"].isin(valid_riders)]
    trips_clean    = trips_clean[trips_clean["driver_id"].isin(valid_drivers)]
    sessions_clean = sessions_clean[sessions_clean["rider_id"].isin(valid_riders)]

    print(f"Trips dropped    : {before_trips    - len(trips_clean):,}")
    print(f"Sessions dropped : {before_sessions - len(sessions_clean):,}")

    return trips_clean, sessions_clean


# ── Main pipeline ────────────────────────────────────────────
def run_preprocessing(data_raw: str = DATA_RAW,
                      data_processed: str = DATA_PROCESSED) -> dict:
    """
    Full preprocessing pipeline.
    Returns a dictionary of cleaned DataFrames and saves them to disk.
    """
    print("=" * 50)
    print("RideWise — Data Preprocessing")
    print("=" * 50)

    # Load
    riders     = pd.read_csv(os.path.join(data_raw, "riders.csv"),
                             parse_dates=["signup_date"])
    trips      = pd.read_csv(os.path.join(data_raw, "trips.csv"))
    drivers    = pd.read_csv(os.path.join(data_raw, "drivers.csv"),
                             parse_dates=["signup_date"])
    sessions   = pd.read_csv(os.path.join(data_raw, "sessions.csv"))

    # Clean
    riders_clean   = clean_riders(riders)
    trips_clean    = clean_trips(trips)
    drivers_clean  = clean_drivers(drivers)
    sessions_clean = clean_sessions(sessions)

    # Referential integrity
    trips_clean, sessions_clean = check_referential_integrity(
        riders_clean, drivers_clean, trips_clean, sessions_clean
    )

    # Validate
    print(f"\n{'Dataset':<12}  {'Raw':>8}  {'Clean':>8}  {'Dropped':>8}")
    print("-" * 42)
    for name, raw, clean in [
        ("riders",   riders,   riders_clean),
        ("drivers",  drivers,  drivers_clean),
        ("trips",    trips,    trips_clean),
        ("sessions", sessions, sessions_clean),
    ]:
        print(f"{name:<12}  {len(raw):>8,}  {len(clean):>8,}  {len(raw)-len(clean):>8,}")

    # Save
    cleaned = {
        "riders_clean":   riders_clean,
        "trips_clean":    trips_clean,
        "drivers_clean":  drivers_clean,
        "sessions_clean": sessions_clean,
    }
    for fname, df in cleaned.items():
        path = os.path.join(data_processed, f"{fname}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {fname}.csv  ({len(df):,} rows)")

    print("\nPreprocessing complete ✓")
    return cleaned


if __name__ == "__main__":
    run_preprocessing()
