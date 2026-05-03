"""
Tests for src/data/preprocessing.py
=====================================
Run with:
    pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import (
    clean_riders,
    clean_trips,
    clean_drivers,
    clean_sessions,
    check_referential_integrity,
)


# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def sample_riders():
    return pd.DataFrame({
        "user_id":        ["R001", "R002", "R003"],
        "signup_date":    ["2024-01-01", "2024-02-01", "2024-03-01"],
        "age":            [25.7, 33.2, np.nan],
        "loyalty_status": ["Bronze", "Gold", "Silver"],
        "city":           ["Lagos", "Cairo", "Nairobi"],
        "avg_rating_given": [4.5, 3.8, 4.9],
        "churn_prob":     [0.2, 0.8, 0.55],
        "referred_by":    ["R000", np.nan, np.nan],
    })


@pytest.fixture
def sample_trips():
    return pd.DataFrame({
        "trip_id":         ["T001", "T002", "T003"],
        "user_id":         ["R001", "R002", "R001"],
        "driver_id":       ["D001", "D001", "D002"],
        "fare":            [10.0, 20.0, 15.0],
        "surge_multiplier":[1.0,  1.5,  1.0],
        "tip":             [1.0,  np.nan, 2.0],
        "payment_type":    ["Card", "Cash", "Card"],
        "pickup_time":     ["2025-01-10 08:00:00+00:00",
                            "2025-01-11 18:00:00+00:00",
                            "2025-01-12 09:00:00+00:00"],
        "dropoff_time":    ["2025-01-10 08:30:00+00:00",
                            "2025-01-11 18:45:00+00:00",
                            "2025-01-12 09:20:00+00:00"],
        "weather":         ["Sunny", "Rainy", "Cloudy"],
        "city":            ["Lagos", "Cairo", "Lagos"],
        "loyalty_status":  ["Bronze", "Gold", "Bronze"],
        "pickup_lat":  [6.5, 30.1, 6.5],
        "pickup_lng":  [3.4, 31.2, 3.4],
        "dropoff_lat": [6.6, 30.2, 6.6],
        "dropoff_lng": [3.5, 31.3, 3.5],
    })


@pytest.fixture
def sample_drivers():
    return pd.DataFrame({
        "driver_id":       ["D001", "D002"],
        "rating":          [4.5, np.nan],
        "vehicle_type":    ["Sedan", "SUV"],
        "signup_date":     ["2023-06-01", "2023-08-01"],
        "last_active":     ["2025-01-10", "2025-01-12"],
        "city":            ["Lagos", "Lagos"],
        "acceptance_rate": [0.85, np.nan],
    })


@pytest.fixture
def sample_sessions():
    return pd.DataFrame({
        "session_id":    ["S001", "S002"],
        "rider_id":      ["R001", "R002"],
        "session_time":  ["2025-01-10 10:00:00+00:00",
                          "2025-01-11 20:00:00+00:00"],
        "time_on_app":   [300, 600],
        "pages_visited": [5, 8],
        "converted":     [1, 0],
        "city":          ["Lagos", "Cairo"],
        "loyalty_status":["Bronze", "Gold"],
    })


# ── Tests: clean_riders ──────────────────────────────────────
class TestCleanRiders:

    def test_age_rounded_to_integer(self, sample_riders):
        result = clean_riders(sample_riders)
        assert result["age"].dtype.name in ("Int64", "int64", "int32")

    def test_churned_column_created(self, sample_riders):
        result = clean_riders(sample_riders)
        assert "churned" in result.columns

    def test_churn_prob_column_removed(self, sample_riders):
        result = clean_riders(sample_riders)
        assert "churn_prob" not in result.columns

    def test_churned_binary_values(self, sample_riders):
        result = clean_riders(sample_riders)
        assert set(result["churned"].unique()).issubset({0, 1})

    def test_churned_threshold_correct(self, sample_riders):
        result = clean_riders(sample_riders)
        # R001 churn_prob=0.2 → 0, R002=0.8 → 1, R003=0.55 → 1
        assert result.loc[result["user_id"] == "R001", "churned"].values[0] == 0
        assert result.loc[result["user_id"] == "R002", "churned"].values[0] == 1

    def test_was_referred_column_created(self, sample_riders):
        result = clean_riders(sample_riders)
        assert "was_referred" in result.columns

    def test_referred_by_column_removed(self, sample_riders):
        result = clean_riders(sample_riders)
        assert "referred_by" not in result.columns

    def test_was_referred_binary(self, sample_riders):
        result = clean_riders(sample_riders)
        # R001 was referred, R002 and R003 were not
        assert result.loc[result["user_id"] == "R001", "was_referred"].values[0] == 1
        assert result.loc[result["user_id"] == "R002", "was_referred"].values[0] == 0


# ── Tests: clean_trips ───────────────────────────────────────
class TestCleanTrips:

    def test_timestamps_parsed(self, sample_trips):
        result = clean_trips(sample_trips)
        assert pd.api.types.is_datetime64_any_dtype(result["pickup_time"])
        assert pd.api.types.is_datetime64_any_dtype(result["dropoff_time"])

    def test_no_timezone_in_timestamps(self, sample_trips):
        result = clean_trips(sample_trips)
        assert result["pickup_time"].dt.tz is None

    def test_trip_duration_positive(self, sample_trips):
        result = clean_trips(sample_trips)
        assert (result["trip_duration"] > 0).all()

    def test_total_revenue_calculated(self, sample_trips):
        result = clean_trips(sample_trips)
        assert "total_revenue" in result.columns
        assert (result["total_revenue"] > 0).all()

    def test_tip_nulls_filled(self, sample_trips):
        result = clean_trips(sample_trips)
        assert result["tip"].isnull().sum() == 0

    def test_hour_of_day_range(self, sample_trips):
        result = clean_trips(sample_trips)
        assert result["hour_of_day"].between(0, 23).all()

    def test_day_of_week_valid(self, sample_trips):
        days = {"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"}
        result = clean_trips(sample_trips)
        assert set(result["day_of_week"].unique()).issubset(days)


# ── Tests: clean_drivers ─────────────────────────────────────
class TestCleanDrivers:

    def test_rating_nulls_filled(self, sample_drivers):
        result = clean_drivers(sample_drivers)
        assert result["rating"].isnull().sum() == 0

    def test_acceptance_rate_nulls_filled(self, sample_drivers):
        result = clean_drivers(sample_drivers)
        assert result["acceptance_rate"].isnull().sum() == 0

    def test_row_count_unchanged(self, sample_drivers):
        result = clean_drivers(sample_drivers)
        assert len(result) == len(sample_drivers)


# ── Tests: clean_sessions ────────────────────────────────────
class TestCleanSessions:

    def test_session_time_parsed(self, sample_sessions):
        result = clean_sessions(sample_sessions)
        assert pd.api.types.is_datetime64_any_dtype(result["session_time"])

    def test_no_timezone_in_session_time(self, sample_sessions):
        result = clean_sessions(sample_sessions)
        assert result["session_time"].dt.tz is None


# ── Tests: referential integrity ─────────────────────────────
class TestReferentialIntegrity:

    def test_orphan_trips_removed(self, sample_riders, sample_trips,
                                  sample_drivers, sample_sessions):
        riders_clean   = clean_riders(sample_riders)
        drivers_clean  = clean_drivers(sample_drivers)
        trips_clean    = clean_trips(sample_trips)
        sessions_clean = clean_sessions(sample_sessions)

        # Add an orphan trip with unknown user_id
        orphan = trips_clean.iloc[[0]].copy()
        orphan["user_id"] = "R999"
        trips_with_orphan = pd.concat([trips_clean, orphan], ignore_index=True)

        cleaned_trips, _ = check_referential_integrity(
            riders_clean, drivers_clean, trips_with_orphan, sessions_clean
        )
        assert "R999" not in cleaned_trips["user_id"].values
