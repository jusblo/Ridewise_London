"""
RideWise — Customer Segmentation
==================================
K-Means clustering on the master feature table.
Assigns business labels (Champions, Loyal Riders, At-Risk, Dormant)
and saves segmented data + model artefacts.

Usage:
    python -m src.models.segmentation
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
FIGURES_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Colour palette ───────────────────────────────────────────
PALETTE = ["#2D6A4F", "#52B788", "#E9C46A", "#E76F51"]
RED     = "#E76F51"

sns.set_theme(style="whitegrid", font_scale=1.2)

# ── Clustering features ──────────────────────────────────────
CLUSTER_COLS = [
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
    "avg_fare", "avg_surge", "tip_rate", "peak_hour_rate", "weekend_ratio",
    "session_last_30d", "avg_time_on_app", "session_conversion_rate", "engagement_score",
    "account_age_days", "activity_trend_30d",
]


# ── Core functions ───────────────────────────────────────────
def prepare_clustering_data(features: pd.DataFrame) -> tuple:
    """Drop NaNs, scale features, return (seg_df, X_scaled)."""
    seg_df = features[["user_id", "churned"] + CLUSTER_COLS].dropna().copy()
    print(f"Riders for segmentation: {len(seg_df):,}")

    scaler = StandardScaler()
    X      = scaler.fit_transform(seg_df[CLUSTER_COLS])
    return seg_df, X, scaler


def find_optimal_k(X: np.ndarray,
                   k_range: range = range(2, 9)) -> tuple:
    """Run elbow + silhouette analysis and return lists of scores."""
    inertias, silhouettes = [], []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels,
                                            sample_size=3000, random_state=42))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(list(k_range), inertias, "o-", color="#2D6A4F", linewidth=2, markersize=7)
    axes[0].axvline(4, color=RED, linestyle="--", label="k = 4 (chosen)")
    axes[0].set(title="Elbow Curve (Inertia)", xlabel="k", ylabel="Inertia")
    axes[0].legend()

    axes[1].plot(list(k_range), silhouettes, "o-", color="#E9C46A", linewidth=2, markersize=7)
    axes[1].axvline(4, color=RED, linestyle="--", label="k = 4 (chosen)")
    axes[1].set(title="Silhouette Score", xlabel="k", ylabel="Score")
    axes[1].legend()

    plt.suptitle("Choosing the Optimal Number of Clusters",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "elbow_silhouette.png"), bbox_inches="tight")
    plt.show()
    print("Saved elbow_silhouette.png")

    return inertias, silhouettes


def fit_kmeans(X: np.ndarray, k: int = 4) -> KMeans:
    """Fit final K-Means model."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    sil = silhouette_score(X, kmeans.labels_, sample_size=3000, random_state=42)
    print(f"k={k} | Inertia: {kmeans.inertia_:,.0f} | Silhouette: {sil:.3f}")
    return kmeans


def assign_segment_labels(seg_df: pd.DataFrame) -> dict:
    """
    Rank clusters by average RFM score and map to business labels:
    Champions, Loyal Riders, At-Risk, Dormant.
    """
    rfm_mean = seg_df.groupby("cluster")[
        ["rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score"]
    ].mean().mean(axis=1)

    rank   = rfm_mean.rank(ascending=False).astype(int)
    names  = ["Champions", "Loyal Riders", "At-Risk", "Dormant"]
    labels = {cluster_id: names[r - 1] for cluster_id, r in rank.items()}

    seg_df["segment"] = seg_df["cluster"].map(labels)
    return labels


def plot_segment_results(seg_df: pd.DataFrame) -> None:
    """Bar charts showing segment sizes and churn rates."""
    segment_order = ["Champions", "Loyal Riders", "At-Risk", "Dormant"]
    fig, axes     = plt.subplots(1, 2, figsize=(14, 5))

    sizes = seg_df["segment"].value_counts().reindex(segment_order)
    bars  = axes[0].bar(sizes.index, sizes.values, color=PALETTE, edgecolor="white")
    for bar, v in zip(bars, sizes.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 30,
                     f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set(title="Riders per Segment", xlabel="Segment", ylabel="Count")
    axes[0].tick_params(axis="x", rotation=15)

    churn_seg = (seg_df.groupby("segment")["churned"].mean()
                 .mul(100).reindex(segment_order))
    bars2 = axes[1].bar(churn_seg.index, churn_seg.values, color=PALETTE, edgecolor="white")
    for bar, v in zip(bars2, churn_seg.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1].set(title="Churn Rate by Segment", xlabel="Segment", ylabel="Churn Rate (%)")
    axes[1].tick_params(axis="x", rotation=15)

    plt.suptitle("Customer Segmentation Results", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "segment_results.png"), bbox_inches="tight")
    plt.show()
    print("Saved segment_results.png")


# ── Main pipeline ────────────────────────────────────────────
def run_segmentation(data_processed: str = DATA_PROCESSED,
                     model_dir: str = MODEL_DIR,
                     k: int = 4) -> pd.DataFrame:
    """Full segmentation pipeline."""
    print("=" * 50)
    print("RideWise — Customer Segmentation")
    print("=" * 50)

    features = pd.read_csv(os.path.join(data_processed, "features.csv"))
    print(f"Features loaded: {features.shape}")

    seg_df, X, scaler = prepare_clustering_data(features)
    find_optimal_k(X)

    kmeans            = fit_kmeans(X, k=k)
    seg_df["cluster"] = kmeans.labels_
    segment_labels    = assign_segment_labels(seg_df)

    print("\nSegment distribution:")
    print(seg_df["segment"].value_counts())
    print("\nChurn rate by segment:")
    print(seg_df.groupby("segment")["churned"].mean()
          .mul(100).round(1).sort_values(ascending=False))

    plot_segment_results(seg_df)

    # Save model artefacts
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("\nModel and scaler saved ✓")

    # Save segmented data
    out = features.merge(seg_df[["user_id", "cluster", "segment"]],
                         on="user_id", how="left")
    out_path = os.path.join(data_processed, "features_segmented.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved features_segmented.csv  ({len(out):,} rows × {out.shape[1]} cols)")
    print(f"\nSegment label map: {segment_labels}")
    print("\nSegmentation complete ✓")
    return out


if __name__ == "__main__":
    run_segmentation()
