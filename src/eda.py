"""
RideWise — Exploratory Data Analysis
=====================================
Loads all raw datasets, runs basic quality checks, and saves
summary visualisations to outputs/figures/.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Colour palette ───────────────────────────────────────────
PALETTE = ["#2D6A4F", "#52B788", "#95D5B2", "#D8F3DC", "#B7E4C7"]
RED     = "#E76F51"
ORANGE  = "#E9C46A"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Data loading ─────────────────────────────────────────────
def load_data(data_raw: str = DATA_RAW) -> dict:
    """Load all raw CSV files and return as a dictionary of DataFrames."""
    datasets = {
        "riders":     pd.read_csv(os.path.join(data_raw, "riders.csv"),
                                  parse_dates=["signup_date"]),
        "trips":      pd.read_csv(os.path.join(data_raw, "trips.csv")),
        "drivers":    pd.read_csv(os.path.join(data_raw, "drivers.csv"),
                                  parse_dates=["signup_date"]),
        "sessions":   pd.read_csv(os.path.join(data_raw, "sessions.csv")),
        "promotions": pd.read_csv(os.path.join(data_raw, "promotions.csv")),
    }
    for name, df in datasets.items():
        print(f"  {name:<12} {df.shape[0]:>7,} rows  {df.shape[1]:>2} cols")
    return datasets


# ── Quality checks ───────────────────────────────────────────
def data_quality_report(datasets: dict) -> None:
    """Print null counts for each dataset."""
    for name, df in datasets.items():
        nulls = df.isnull().sum()
        if nulls.any():
            print(f"\n── {name} nulls ──")
            print(nulls[nulls > 0])
        else:
            print(f"\n── {name}: no nulls ──")


# ── Visualisations ───────────────────────────────────────────
def plot_rider_demographics(riders: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Age
    axes[0].hist(riders["age"].dropna(), bins=30, color=PALETTE[0], edgecolor="white")
    axes[0].axvline(riders["age"].mean(), color=RED, linestyle="--",
                    label=f'Mean: {riders["age"].mean():.1f}')
    axes[0].set(title="Age Distribution", xlabel="Age", ylabel="Riders")
    axes[0].legend()

    # City
    city_counts = riders["city"].value_counts()
    bars = axes[1].bar(city_counts.index, city_counts.values,
                       color=PALETTE[:len(city_counts)], edgecolor="white")
    for bar, v in zip(bars, city_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 40,
                     f"{v:,}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set(title="Riders by City", xlabel="City", ylabel="Count")

    # Loyalty
    order = ["Bronze", "Silver", "Gold", "Platinum"]
    loy   = riders["loyalty_status"].value_counts().reindex(order)
    bars2 = axes[2].bar(loy.index, loy.values,
                        color=["#CD7F32", "#C0C0C0", "#FFD700", "#E5E4E2"],
                        edgecolor="white")
    for bar, v in zip(bars2, loy.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 40,
                     f"{v:,}", ha="center", fontsize=9, fontweight="bold")
    axes[2].set(title="Riders by Loyalty Tier", xlabel="Tier", ylabel="Count")

    plt.suptitle("Rider Demographics Overview", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "rider_demographics.png"), bbox_inches="tight")
    plt.show()
    print("Saved rider_demographics.png")


def plot_churn_distribution(riders: pd.DataFrame) -> None:
    riders["churned"] = (riders["churn_prob"] >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(riders["churn_prob"].dropna(), bins=40,
                 color=PALETTE[1], edgecolor="white")
    axes[0].axvline(0.5, color=RED, linestyle="--", label="Decision boundary (0.5)")
    axes[0].set(title="Churn Probability Distribution",
                xlabel="Churn Probability", ylabel="Riders")
    axes[0].legend()

    city_churn = riders.groupby("city")["churned"].mean().mul(100)
    bars = axes[1].bar(city_churn.index, city_churn.values,
                       color=PALETTE[:len(city_churn)], edgecolor="white")
    for bar, v in zip(bars, city_churn.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set(title="Churn Rate by City", xlabel="City", ylabel="Churn Rate (%)")

    plt.suptitle("Churn Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "churn_distribution.png"), bbox_inches="tight")
    plt.show()
    print("Saved churn_distribution.png")


def plot_trip_patterns(trips: pd.DataFrame) -> None:
    for col in ["pickup_time", "dropoff_time"]:
        trips[col] = pd.to_datetime(trips[col], utc=True).dt.tz_convert(None)

    trips = trips.dropna(subset=["pickup_time", "dropoff_time"])
    trips["hour_of_day"] = trips["pickup_time"].dt.hour
    trips["day_of_week"] = trips["pickup_time"].dt.day_name()
    trips = trips[
        (trips["dropoff_time"] - trips["pickup_time"]).dt.total_seconds() / 60 > 0
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    hour_counts = trips["hour_of_day"].value_counts().sort_index()
    axes[0].bar(hour_counts.index, hour_counts.values, color=PALETTE[1], edgecolor="white")
    axes[0].axvspan(6.5,  9.5,  alpha=0.18, color=ORANGE, label="Rush hour")
    axes[0].axvspan(16.5, 19.5, alpha=0.18, color=ORANGE)
    axes[0].set(title="Trips by Hour of Day", xlabel="Hour", ylabel="Trips")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()

    day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_colors = [PALETTE[0]] * 5 + [RED, RED]
    day_counts = trips["day_of_week"].value_counts().reindex(day_order)
    axes[1].bar(day_order, day_counts.values, color=day_colors, edgecolor="white")
    axes[1].set(title="Trips by Day of Week", xlabel="Day", ylabel="Trips")
    axes[1].tick_params(axis="x", rotation=35)

    plt.suptitle("Trip Timing Patterns", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "trip_patterns.png"), bbox_inches="tight")
    plt.show()
    print("Saved trip_patterns.png")


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("RideWise — Exploratory Data Analysis")
    print("=" * 50)

    datasets = load_data()
    data_quality_report(datasets)

    riders = datasets["riders"]
    trips  = datasets["trips"]

    plot_rider_demographics(riders)
    plot_churn_distribution(riders)
    plot_trip_patterns(trips)

    print("\nEDA complete ✓")


if __name__ == "__main__":
    main()
