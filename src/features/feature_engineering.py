"""
RideWise — Feature Engineering
================================
Builds the master feature table from cleaned datasets.

Features engineered:
  - RFM scores (Recency, Frequency, Monetary)
  - Behavioural trip features
  - Session engagement features
  - Activity trend signal
  - Composite engagement score

Usage:
    python -m src.features.feature_engineering
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")


# ── Eligibility filter ───────────────────────────────────────
def get_eligible_riders(riders: pd.DataFrame,
                        trips: pd.DataFrame,
                        reference_date: pd.Timestamp) -> pd.DataFrame:
    """
    Return riders with at least 1 trip and account age >= 60 days.
    """
    riders = riders.copy()
    riders["account_age_days"] = (reference_date - riders["signup_date"]).dt.days

    riders_with_trips = set(trips["user_id"])
    eligible = riders[
        riders["user_id"].isin(riders_with_trips) &
        (riders["account_age_days"] >= 60)
    ][["user_id", "churned"]].copy()

    print(f"Total riders   : {len(riders):,}")
    print(f"Eligible riders: {len(eligible):,}")
    print(f"Churn rate     : {eligible['churned'].mean() * 100:.1f}%")
    return eligible


# ── RFM features ─────────────────────────────────────────────
def build_recency(eligible: pd.DataFrame,
                  trips: pd.DataFrame,
                  sessions: pd.DataFrame,
                  reference_date: pd.Timestamp) -> pd.DataFrame:
    """Days since last trip and last app session."""
    last_trip = (trips.groupby("user_id")["pickup_time"].max()
                 .reset_index(name="last_trip"))
    last_session = (sessions.groupby("rider_id")["session_time"].max()
                    .reset_index()
                    .rename(columns={"session_time": "last_session",
                                     "rider_id": "user_id"}))

    recency = (eligible[["user_id"]]
               .merge(last_trip,    on="user_id", how="left")
               .merge(last_session, on="user_id", how="left"))

    recency["days_since_last_trip"]    = (
        reference_date - recency["last_trip"]).dt.days.fillna(999)
    recency["days_since_last_session"] = (
        reference_date - recency["last_session"]).dt.days.fillna(999)

    return recency[["user_id", "days_since_last_trip", "days_since_last_session"]]


def build_frequency(eligible: pd.DataFrame,
                    trips: pd.DataFrame,
                    reference_date: pd.Timestamp) -> pd.DataFrame:
    """Trip counts over rolling 7 / 30 / 60 / 90-day windows."""
    frequency = eligible[["user_id"]].copy()

    for days in [7, 30, 60, 90]:
        cutoff = reference_date - pd.Timedelta(days=days)
        counts = (trips[trips["pickup_time"] >= cutoff]
                  .groupby("user_id").size()
                  .reset_index(name=f"trips_last_{days}d"))
        frequency = frequency.merge(counts, on="user_id", how="left")

    freq_cols = [f"trips_last_{d}d" for d in [7, 30, 60, 90]]
    frequency[freq_cols] = frequency[freq_cols].fillna(0).astype(int)
    return frequency


def build_monetary(trips: pd.DataFrame,
                   reference_date: pd.Timestamp) -> pd.DataFrame:
    """Lifetime and recent (30-day) revenue metrics."""
    monetary = (trips.groupby("user_id")
                .agg(monetary_total  =("total_revenue", "sum"),
                     monetary_avg    =("total_revenue", "mean"),
                     trips_lifetime  =("trip_id",       "count"))
                .round(2).reset_index())

    cutoff_30   = reference_date - pd.Timedelta(days=30)
    monetary_30 = (trips[trips["pickup_time"] >= cutoff_30]
                   .groupby("user_id")["total_revenue"].sum()
                   .reset_index(name="monetary_last_30d"))

    monetary = monetary.merge(monetary_30, on="user_id", how="left")
    monetary["monetary_last_30d"] = monetary["monetary_last_30d"].fillna(0)
    return monetary


def build_rfm_scores(recency: pd.DataFrame,
                     frequency: pd.DataFrame,
                     monetary: pd.DataFrame) -> pd.DataFrame:
    """Quantile-bin R, F, M into 1–5 scores and compute combined score."""
    rfm = (recency[["user_id", "days_since_last_trip"]]
           .merge(frequency[["user_id", "trips_last_30d"]], on="user_id", how="left")
           .merge(monetary[["user_id", "monetary_total"]],  on="user_id", how="left")
           .fillna({"trips_last_30d": 0, "monetary_total": 0}))

    # Lower days = better → invert labels
    rfm["rfm_recency_score"]   = pd.qcut(
        rfm["days_since_last_trip"].rank(method="first"),
        q=5, labels=[5, 4, 3, 2, 1]).astype(int)

    rfm["rfm_frequency_score"] = pd.qcut(
        rfm["trips_last_30d"].rank(method="first"),
        q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["rfm_monetary_score"]  = pd.qcut(
        rfm["monetary_total"].rank(method="first"),
        q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Combined score: average of the three RFM SCORES (not raw values)
    rfm["rfm_combined_score"]  = (
        (rfm["rfm_recency_score"] +
         rfm["rfm_frequency_score"] +
         rfm["rfm_monetary_score"]) / 3
    ).round(2)

    return rfm


# ── Behavioural & session features ───────────────────────────
def build_trip_stats(trips: pd.DataFrame) -> pd.DataFrame:
    """Per-rider trip behaviour aggregates."""
    return (trips.groupby("user_id").agg(
        avg_fare       =("fare",            "mean"),
        avg_surge      =("surge_multiplier","mean"),
        tip_rate       =("tip",             lambda x: (x > 0).mean()),
        peak_hour_rate =("hour_of_day",     lambda x: x.isin([7,8,9,17,18,19]).mean()),
        weekend_ratio  =("day_of_week",     lambda x: x.isin(["Saturday","Sunday"]).mean()),
    ).round(3).reset_index())


def build_session_stats(sessions: pd.DataFrame,
                        reference_date: pd.Timestamp) -> pd.DataFrame:
    """Per-rider session engagement aggregates."""
    cutoff_30 = reference_date - pd.Timedelta(days=30)
    return (sessions.groupby("rider_id").agg(
        session_last_30d        =("session_time", lambda x: (x >= cutoff_30).sum()),
        avg_time_on_app         =("time_on_app",  "mean"),
        session_conversion_rate =("converted",    "mean"),
    ).round(3).reset_index().rename(columns={"rider_id": "user_id"}))


def build_activity_trend(frequency: pd.DataFrame) -> pd.DataFrame:
    """Ratio of last-30d trips to last-60d trips (trend signal)."""
    trend = frequency[["user_id", "trips_last_30d", "trips_last_60d"]].copy()
    trend["activity_trend_30d"] = (
        trend["trips_last_30d"] / (trend["trips_last_60d"] + 1)
    ).round(3)
    return trend[["user_id", "activity_trend_30d"]]


# ── Master feature table ─────────────────────────────────────
def build_feature_table(eligible, recency, frequency, monetary,
                        trip_stats, session_stats, activity, rfm,
                        riders) -> pd.DataFrame:
    """Merge all feature groups into one table."""
    features = eligible.copy()
    features = features.merge(
        recency[["user_id", "days_since_last_trip", "days_since_last_session"]],
        on="user_id", how="left")
    features = features.merge(frequency,               on="user_id", how="left")
    features = features.merge(monetary,                on="user_id", how="left")
    features = features.merge(trip_stats,              on="user_id", how="left")
    features = features.merge(session_stats,           on="user_id", how="left")
    features = features.merge(activity,                on="user_id", how="left")
    features = features.merge(
        rfm[["user_id", "rfm_recency_score", "rfm_frequency_score",
             "rfm_monetary_score", "rfm_combined_score"]],
        on="user_id", how="left")
    features = features.merge(
        riders[["user_id", "account_age_days", "was_referred",
                "loyalty_status", "age", "city"]],
        on="user_id", how="left")

    # Composite engagement score (weighted RFM)
    features["engagement_score"] = (
        features["rfm_recency_score"]   * 0.40 +
        features["rfm_frequency_score"] * 0.35 +
        features["rfm_monetary_score"]  * 0.25
    ).round(2)

    return features


# ── Main pipeline ────────────────────────────────────────────
def run_feature_engineering(data_processed: str = DATA_PROCESSED) -> pd.DataFrame:
    """Full feature engineering pipeline. Returns feature DataFrame."""
    print("=" * 50)
    print("RideWise — Feature Engineering")
    print("=" * 50)

    riders   = pd.read_csv(os.path.join(data_processed, "riders_clean.csv"),
                           parse_dates=["signup_date"])
    trips    = pd.read_csv(os.path.join(data_processed, "trips_clean.csv"),
                           parse_dates=["pickup_time", "dropoff_time"])
    sessions = pd.read_csv(os.path.join(data_processed, "sessions_clean.csv"),
                           parse_dates=["session_time"])

    reference_date = trips["pickup_time"].max()
    print(f"Reference date : {reference_date.date()}")

    eligible      = get_eligible_riders(riders, trips, reference_date)
    recency       = build_recency(eligible, trips, sessions, reference_date)
    frequency     = build_frequency(eligible, trips, reference_date)
    monetary      = build_monetary(trips, reference_date)
    rfm           = build_rfm_scores(recency, frequency, monetary)
    trip_stats    = build_trip_stats(trips)
    session_stats = build_session_stats(sessions, reference_date)
    activity      = build_activity_trend(frequency)

    riders["account_age_days"] = (reference_date - riders["signup_date"]).dt.days

    features = build_feature_table(
        eligible, recency, frequency, monetary,
        trip_stats, session_stats, activity, rfm, riders
    )

    out_path = os.path.join(data_processed, "features.csv")
    features.to_csv(out_path, index=False)
    print(f"\nSaved features.csv  ({len(features):,} rows × {features.shape[1]} cols)")
    print("\nFeature engineering complete ✓")
    return features


if __name__ == "__main__":
    run_feature_engineering()
