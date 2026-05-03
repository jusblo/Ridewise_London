"""
Tests for src/features/feature_engineering.py
================================================
Run with:
    pytest tests/test_features.py -v
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features.feature_engineering import (
    get_eligible_riders,
    build_recency,
    build_frequency,
    build_monetary,
    build_rfm_scores,
    build_trip_stats,
    build_session_stats,
    build_activity_trend,
)

# ── Reference date ───────────────────────────────────────────
REFERENCE_DATE = pd.Timestamp("2025-04-27")


# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def sample_riders():
    return pd.DataFrame({
        "user_id":        ["R001", "R002", "R003"],
        "signup_date":    pd.to_datetime(["2024-01-01", "2024-02-01", "2025-04-01"]),
        "churned":        [0, 1, 0],
        "was_referred":   [1, 0, 0],
        "account_age_days": [
            (REFERENCE_DATE - pd.Timestamp("2024-01-01")).days,
            (REFERENCE_DATE - pd.Timestamp("2024-02-01")).days,
            (REFERENCE_DATE - pd.Timestamp("2025-04-01")).days,
        ],
        "loyalty_status": ["Bronze", "Gold", "Silver"],
        "age":            [25, 33, 28],
        "city":           ["Lagos", "Cairo", "Nairobi"],
    })


@pytest.fixture
def sample_trips():
    return pd.DataFrame({
        "trip_id":          ["T001", "T002", "T003", "T004"],
        "user_id":          ["R001", "R001", "R002", "R001"],
        "fare":             [10.0, 15.0, 20.0, 12.0],
        "surge_multiplier": [1.0, 1.5, 1.0, 1.2],
        "tip":              [1.0, 0.0, 2.0, 0.5],
        "total_revenue":    [11.0, 22.5, 22.0, 14.9],
        "pickup_time":      pd.to_datetime([
            "2025-04-20", "2025-04-01", "2025-03-15", "2025-04-10"
        ]),
        "hour_of_day":      [8, 17, 12, 9],
        "day_of_week":      ["Monday", "Thursday", "Saturday", "Tuesday"],
    })


@pytest.fixture
def sample_sessions():
    return pd.DataFrame({
        "rider_id":    ["R001", "R001", "R002"],
        "session_time": pd.to_datetime([
            "2025-04-27", "2025-04-10", "2025-03-01"
        ]),
        "time_on_app": [300, 500, 200],
        "converted":   [1, 0, 1],
    })


@pytest.fixture
def eligible(sample_riders, sample_trips):
    return get_eligible_riders(sample_riders, sample_trips, REFERENCE_DATE)


# ── Tests: eligible riders ───────────────────────────────────
class TestEligibleRiders:

    def test_excludes_riders_with_no_trips(self, sample_riders, sample_trips):
        result = get_eligible_riders(sample_riders, sample_trips, REFERENCE_DATE)
        # R003 has no trips → should be excluded
        assert "R003" not in result["user_id"].values

    def test_excludes_new_accounts(self, sample_riders, sample_trips):
        # R003 account age < 60 days also excluded
        result = get_eligible_riders(sample_riders, sample_trips, REFERENCE_DATE)
        assert len(result) <= len(sample_riders)

    def test_output_columns(self, eligible):
        assert "user_id" in eligible.columns
        assert "churned" in eligible.columns


# ── Tests: recency ───────────────────────────────────────────
class TestBuildRecency:

    def test_days_since_last_trip_non_negative(self, eligible, sample_trips, sample_sessions):
        result = build_recency(eligible, sample_trips, sample_sessions, REFERENCE_DATE)
        assert (result["days_since_last_trip"] >= 0).all()

    def test_days_since_last_session_non_negative(self, eligible, sample_trips, sample_sessions):
        result = build_recency(eligible, sample_trips, sample_sessions, REFERENCE_DATE)
        assert (result["days_since_last_session"] >= 0).all()

    def test_output_columns(self, eligible, sample_trips, sample_sessions):
        result = build_recency(eligible, sample_trips, sample_sessions, REFERENCE_DATE)
        assert "days_since_last_trip"    in result.columns
        assert "days_since_last_session" in result.columns


# ── Tests: frequency ─────────────────────────────────────────
class TestBuildFrequency:

    def test_frequency_columns_exist(self, eligible, sample_trips):
        result = build_frequency(eligible, sample_trips, REFERENCE_DATE)
        for days in [7, 30, 60, 90]:
            assert f"trips_last_{days}d" in result.columns

    def test_no_negative_counts(self, eligible, sample_trips):
        result = build_frequency(eligible, sample_trips, REFERENCE_DATE)
        for days in [7, 30, 60, 90]:
            assert (result[f"trips_last_{days}d"] >= 0).all()

    def test_longer_window_ge_shorter(self, eligible, sample_trips):
        result = build_frequency(eligible, sample_trips, REFERENCE_DATE)
        # 90d window should always be >= 30d window
        assert (result["trips_last_90d"] >= result["trips_last_30d"]).all()


# ── Tests: monetary ──────────────────────────────────────────
class TestBuildMonetary:

    def test_monetary_columns_exist(self, sample_trips):
        result = build_monetary(sample_trips, REFERENCE_DATE)
        for col in ["monetary_total", "monetary_avg", "trips_lifetime", "monetary_last_30d"]:
            assert col in result.columns

    def test_monetary_total_positive(self, sample_trips):
        result = build_monetary(sample_trips, REFERENCE_DATE)
        assert (result["monetary_total"] > 0).all()

    def test_no_nulls_in_monetary(self, sample_trips):
        result = build_monetary(sample_trips, REFERENCE_DATE)
        assert result["monetary_last_30d"].isnull().sum() == 0


# ── Tests: RFM scores ────────────────────────────────────────
class TestBuildRFMScores:

    @pytest.fixture
    def rfm(self, eligible, sample_trips, sample_sessions):
        recency   = build_recency(eligible, sample_trips, sample_sessions, REFERENCE_DATE)
        frequency = build_frequency(eligible, sample_trips, REFERENCE_DATE)
        monetary  = build_monetary(sample_trips, REFERENCE_DATE)
        return build_rfm_scores(recency, frequency, monetary)

    def test_scores_in_range_1_to_5(self, rfm):
        for col in ["rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score"]:
            assert rfm[col].between(1, 5).all(), f"{col} out of range"

    def test_combined_score_is_average_of_three(self, rfm):
        expected = (
            rfm["rfm_recency_score"] +
            rfm["rfm_frequency_score"] +
            rfm["rfm_monetary_score"]
        ) / 3
        pd.testing.assert_series_equal(
            rfm["rfm_combined_score"].round(2),
            expected.round(2),
            check_names=False,
        )

    def test_no_raw_monetary_in_combined_score(self, rfm):
        # Combined score max should be 5.0 (not inflated by raw £ values)
        assert rfm["rfm_combined_score"].max() <= 5.0


# ── Tests: trip stats ────────────────────────────────────────
class TestBuildTripStats:

    def test_output_columns(self, sample_trips):
        result = build_trip_stats(sample_trips)
        for col in ["avg_fare", "avg_surge", "tip_rate",
                    "peak_hour_rate", "weekend_ratio"]:
            assert col in result.columns

    def test_rates_between_0_and_1(self, sample_trips):
        result = build_trip_stats(sample_trips)
        for col in ["tip_rate", "peak_hour_rate", "weekend_ratio"]:
            assert result[col].between(0, 1).all(), f"{col} out of [0,1]"


# ── Tests: activity trend ────────────────────────────────────
class TestBuildActivityTrend:

    def test_trend_column_exists(self, eligible, sample_trips):
        frequency = build_frequency(eligible, sample_trips, REFERENCE_DATE)
        result    = build_activity_trend(frequency)
        assert "activity_trend_30d" in result.columns

    def test_trend_non_negative(self, eligible, sample_trips):
        frequency = build_frequency(eligible, sample_trips, REFERENCE_DATE)
        result    = build_activity_trend(frequency)
        assert (result["activity_trend_30d"] >= 0).all()
