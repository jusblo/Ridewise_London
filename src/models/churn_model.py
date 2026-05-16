"""
RideWise — Churn Prediction Model
===================================
Trains a Logistic Regression (baseline) and Random Forest (primary)
churn prediction model on the master feature table.

Saves model artefacts to the models/ folder:
  - churn_rf_model.pkl   (Random Forest)
  - churn_lr_model.pkl   (Logistic Regression)
  - churn_scaler.pkl     (StandardScaler for LR)

Saves predictions to data/processed/:
  - churn_predictions.csv

Usage:
    python -m src.models.churn_model
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
FIGURES_DIR    = os.path.join(BASE_DIR, "outputs", "figures")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────
PALETTE = ["#2D6A4F", "#52B788", "#95D5B2", "#D8F3DC", "#B7E4C7"]
RED     = "#E76F51"
ORANGE  = "#E9C46A"

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Feature columns ───────────────────────────────────────────
FEATURE_COLS = [
    # Account & demographics
    "age", "account_age_days", "was_referred",
    "loyalty_encoded", "city_encoded",
    # Trip behaviour
    "n_trips", "avg_fare", "avg_surge", "total_spend", "tip_rate",
    "peak_hour_rate", "weekend_ratio", "days_since_last",
    "trips_last_7d", "trips_last_30d", "trips_last_60d", "trips_last_90d",
    "activity_trend_30d",
    # Session engagement
    "n_sessions", "session_last_30d", "avg_time_on_app",
    "conv_rate", "days_since_session",
    # RFM
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
    "rfm_combined_score", "engagement_score",
]


# ── Data loading & feature engineering ───────────────────────
def load_and_engineer_features(data_raw: str = DATA_RAW) -> pd.DataFrame:
    """
    Load raw CSVs, clean, and engineer all features needed
    for churn prediction. Returns a model-ready DataFrame.
    """
    print("Loading raw data...")
    riders   = pd.read_csv(os.path.join(data_raw, "riders.csv"),
                           parse_dates=["signup_date"])
    trips    = pd.read_csv(os.path.join(data_raw, "trips.csv"))
    sessions = pd.read_csv(os.path.join(data_raw, "sessions.csv"))
    drivers  = pd.read_csv(os.path.join(data_raw, "drivers.csv"))

    # ── Clean trips ───────────────────────────────────────────
    for col in ["pickup_time", "dropoff_time"]:
        trips[col] = pd.to_datetime(trips[col], utc=True).dt.tz_convert(None)
    trips = trips.dropna(subset=["pickup_time", "dropoff_time"])
    trips["trip_duration"] = (
        (trips["dropoff_time"] - trips["pickup_time"]).dt.total_seconds() / 60
    )
    trips["total_revenue"] = (
        trips["fare"] * trips["surge_multiplier"] + trips["tip"].fillna(0)
    )
    trips["hour_of_day"] = trips["pickup_time"].dt.hour
    trips["day_of_week"] = trips["pickup_time"].dt.day_name()
    trips = trips[trips["trip_duration"] > 0]

    # ── Clean sessions ────────────────────────────────────────
    sessions["session_time"] = (
        pd.to_datetime(sessions["session_time"], utc=True).dt.tz_convert(None)
    )
    sessions = sessions.dropna(subset=["session_time"])

    # ── Rider labels ──────────────────────────────────────────
    reference_date             = trips["pickup_time"].max()
    riders["churned"]          = (riders["churn_prob"] > 0.5).astype(int)
    riders["was_referred"]     = riders["referred_by"].notna().astype(int)
    riders["account_age_days"] = (reference_date - riders["signup_date"]).dt.days

    # ── Referential integrity ─────────────────────────────────
    valid_riders  = set(riders["user_id"])
    valid_drivers = set(drivers["driver_id"])
    trips    = trips[trips["user_id"].isin(valid_riders) &
                     trips["driver_id"].isin(valid_drivers)]
    sessions = sessions[sessions["rider_id"].isin(valid_riders)]

    # ── Eligible riders ───────────────────────────────────────
    riders_with_trips = set(trips["user_id"])
    eligible = riders[
        riders["user_id"].isin(riders_with_trips) &
        (riders["account_age_days"] >= 60)
    ].copy()

    print(f"  Eligible riders : {len(eligible):,}")
    print(f"  Churn rate      : {eligible['churned'].mean()*100:.1f}%")

    # ── Trip aggregates ───────────────────────────────────────
    cutoff_30 = reference_date - pd.Timedelta(days=30)

    trip_agg = trips.groupby("user_id").agg(
        n_trips         = ("trip_id",        "count"),
        avg_fare        = ("fare",           "mean"),
        avg_surge       = ("surge_multiplier","mean"),
        total_spend     = ("total_revenue",  "sum"),
        tip_rate        = ("tip",            lambda x: (x > 0).mean()),
        peak_hour_rate  = ("hour_of_day",    lambda x: x.isin([7,8,9,17,18,19]).mean()),
        weekend_ratio   = ("day_of_week",    lambda x: x.isin(["Saturday","Sunday"]).mean()),
        days_since_last = ("pickup_time",    lambda x: (reference_date - x.max()).days),
    ).round(3).reset_index()

    for days in [7, 30, 60, 90]:
        cutoff = reference_date - pd.Timedelta(days=days)
        col    = f"trips_last_{days}d"
        counts = (trips[trips["pickup_time"] >= cutoff]
                  .groupby("user_id").size()
                  .reset_index(name=col))
        trip_agg = trip_agg.merge(counts, on="user_id", how="left")

    freq_cols = [f"trips_last_{d}d" for d in [7, 30, 60, 90]]
    trip_agg[freq_cols] = trip_agg[freq_cols].fillna(0).astype(int)

    trip_agg["activity_trend_30d"] = (
        trip_agg["trips_last_30d"] / (trip_agg["trips_last_60d"] + 1)
    ).round(3)

    # ── Session aggregates ────────────────────────────────────
    session_agg = sessions.groupby("rider_id").agg(
        n_sessions        = ("session_time", "count"),
        session_last_30d  = ("session_time", lambda x: (x >= cutoff_30).sum()),
        avg_time_on_app   = ("time_on_app",  "mean"),
        conv_rate         = ("converted",    "mean"),
        days_since_session= ("session_time", lambda x: (reference_date - x.max()).days),
    ).round(3).reset_index().rename(columns={"rider_id": "user_id"})

    # ── RFM scores ────────────────────────────────────────────
    rfm = trip_agg[["user_id", "days_since_last", "trips_last_30d"]].copy()
    rfm = rfm.merge(
        trips.groupby("user_id")["total_revenue"].sum().reset_index(name="monetary_total"),
        on="user_id", how="left"
    ).fillna({"trips_last_30d": 0, "monetary_total": 0})

    rfm["rfm_recency_score"]   = pd.qcut(
        rfm["days_since_last"].rank(method="first"),
        q=5, labels=[5,4,3,2,1]).astype(int)
    rfm["rfm_frequency_score"] = pd.qcut(
        rfm["trips_last_30d"].rank(method="first"),
        q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_monetary_score"]  = pd.qcut(
        rfm["monetary_total"].rank(method="first"),
        q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_combined_score"]  = (
        (rfm["rfm_recency_score"] +
         rfm["rfm_frequency_score"] +
         rfm["rfm_monetary_score"]) / 3
    ).round(2)
    rfm["engagement_score"] = (
        rfm["rfm_recency_score"]   * 0.40 +
        rfm["rfm_frequency_score"] * 0.35 +
        rfm["rfm_monetary_score"]  * 0.25
    ).round(2)

    # ── Assemble ──────────────────────────────────────────────
    model_df = eligible[["user_id", "churned", "age", "account_age_days",
                          "was_referred", "loyalty_status", "city"]].copy()
    model_df = model_df.merge(trip_agg,    on="user_id", how="left")
    model_df = model_df.merge(session_agg, on="user_id", how="left")
    model_df = model_df.merge(
        rfm[["user_id", "rfm_recency_score", "rfm_frequency_score",
             "rfm_monetary_score", "rfm_combined_score", "engagement_score"]],
        on="user_id", how="left"
    )

    model_df["loyalty_encoded"] = LabelEncoder().fit_transform(
        model_df["loyalty_status"].fillna("Unknown"))
    model_df["city_encoded"]    = LabelEncoder().fit_transform(
        model_df["city"].fillna("Unknown"))

    print(f"  Feature table   : {model_df.shape[0]:,} rows × {model_df.shape[1]} cols")
    return model_df


# ── Model training ────────────────────────────────────────────
def train_models(model_df: pd.DataFrame) -> tuple:
    """
    Train Logistic Regression and Random Forest models.
    Returns (lr, rf, scaler, X_test, y_test, lr_prob, rf_prob).
    """
    X = model_df[FEATURE_COLS].fillna(0)
    y = model_df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set : {len(X_train):,} riders")
    print(f"Test set     : {len(X_test):,} riders")

    # Logistic Regression
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(
        class_weight = "balanced",
        max_iter     = 1000,
        random_state = 42,
        C            = 1.0,
    )
    lr.fit(X_train_s, y_train)
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_pred = lr.predict(X_test_s)

    print("\n=== Logistic Regression ===")
    print(f"  Accuracy  : {accuracy_score(y_test, lr_pred):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, lr_prob):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, lr_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, lr_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, lr_pred):.4f}")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators     = 200,
        max_depth        = 10,
        min_samples_leaf = 20,
        class_weight     = "balanced",
        random_state     = 42,
        n_jobs           = -1,
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)

    print("\n=== Random Forest ===")
    print(f"  Accuracy  : {accuracy_score(y_test, rf_pred):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, rf_prob):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, rf_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, rf_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, rf_pred):.4f}")
    print("\n", classification_report(y_test, rf_pred,
                                      target_names=["Retained", "Churned"]))

    return lr, rf, scaler, X_train, X_test, y_train, y_test, lr_prob, rf_prob


# ── Cross-validation ──────────────────────────────────────────
def cross_validate_models(lr, rf, scaler,
                           X_train: pd.DataFrame,
                           y_train: pd.Series) -> None:
    """Run 5-fold stratified cross-validation for both models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X_train_s = scaler.transform(X_train)
    lr_cv     = cross_val_score(lr, X_train_s, y_train, cv=cv, scoring="roc_auc")
    rf_cv     = cross_val_score(rf, X_train,   y_train, cv=cv, scoring="roc_auc")

    print("\n=== 5-Fold Cross-Validation AUC ===")
    print(f"  Logistic Regression : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
    print(f"  Random Forest       : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    for scores, label, color in [
        (lr_cv, "Logistic Regression", PALETTE[0]),
        (rf_cv, "Random Forest",       RED),
    ]:
        folds = range(1, len(scores) + 1)
        ax.plot(folds, scores, "o-", color=color, linewidth=2,
                markersize=7, label=label)
        ax.axhline(scores.mean(), color=color, linestyle="--", linewidth=1,
                   label=f"{label} mean: {scores.mean():.3f}")

    ax.set(title="5-Fold Cross-Validation AUC-ROC",
           xlabel="Fold", ylabel="AUC-ROC", ylim=(0.3, 0.7))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "churn_cv_auc.png"), bbox_inches="tight")
    plt.show()
    print("Saved churn_cv_auc.png")


# ── Visualisations ────────────────────────────────────────────
def plot_roc_curves(y_test, lr_prob, rf_prob) -> None:
    """Plot ROC curves for both models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for prob, label, color in [
        (lr_prob, "Logistic Regression", PALETTE[0]),
        (rf_prob, "Random Forest",       RED),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc         = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{label}  (AUC = {auc:.3f})")

    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random classifier")
    ax.fill_between(*roc_curve(y_test, rf_prob)[:2], alpha=0.08, color=RED)
    ax.set(title="ROC Curves — Churn Prediction",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "churn_roc_curves.png"), bbox_inches="tight")
    plt.show()
    print("Saved churn_roc_curves.png")


def plot_confusion_matrices(y_test, lr_pred, rf_pred) -> None:
    """Plot confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, pred, title in [
        (axes[0], lr_pred, "Logistic Regression"),
        (axes[1], rf_pred, "Random Forest"),
    ]:
        cm   = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Retained","Churned"])
        disp.plot(ax=ax, colorbar=False, cmap="Greens")
        ax.set_title(title, fontweight="bold")

    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "churn_confusion_matrices.png"),
                bbox_inches="tight")
    plt.show()
    print("Saved churn_confusion_matrices.png")


def plot_feature_importance(rf, top_n: int = 15) -> None:
    """Plot top N feature importances from the Random Forest."""
    fi = (pd.Series(rf.feature_importances_, index=FEATURE_COLS)
          .sort_values(ascending=True)
          .tail(top_n))

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(fi.index, fi.values, color=PALETTE[1], edgecolor="white")
    for bar, v in zip(bars, fi.values):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set(title=f"Top {top_n} Feature Importances — Random Forest",
           xlabel="Importance", ylabel="Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "churn_feature_importance.png"),
                bbox_inches="tight")
    plt.show()
    print("Saved churn_feature_importance.png")


def plot_risk_tiers(model_df: pd.DataFrame) -> None:
    """Plot churn risk score distribution and risk tier breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    risk_probs = model_df["churn_risk_score"]

    axes[0].hist(risk_probs, bins=40, color=PALETTE[1], edgecolor="white")
    axes[0].axvline(0.5, color=RED, linestyle="--", linewidth=1.5,
                    label="Decision threshold (0.5)")
    axes[0].set(title="Churn Risk Score Distribution",
                xlabel="Predicted Churn Probability", ylabel="Riders")
    axes[0].legend()

    tier_counts = (model_df["risk_tier"].value_counts()
                   .reindex(["Low","Medium","High","Critical"]))
    tier_colors = [PALETTE[0], ORANGE, RED, "#9B2226"]
    bars = axes[1].bar(tier_counts.index, tier_counts.values,
                       color=tier_colors, edgecolor="white")
    for bar, v in zip(bars, tier_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 30,
                     f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set(title="Riders by Risk Tier", xlabel="Risk Tier", ylabel="Count")

    plt.suptitle("Churn Risk Scoring", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "churn_risk_tiers.png"), bbox_inches="tight")
    plt.show()
    print("Saved churn_risk_tiers.png")


# ── Risk scoring ──────────────────────────────────────────────
def assign_risk_scores(model_df: pd.DataFrame, rf) -> pd.DataFrame:
    """Assign churn risk score and tier to every eligible rider."""
    X_all      = model_df[FEATURE_COLS].fillna(0)
    risk_probs = rf.predict_proba(X_all)[:, 1]

    model_df = model_df.copy()
    model_df["churn_risk_score"] = risk_probs
    model_df["risk_tier"]        = pd.cut(
        risk_probs,
        bins   = [0, 0.3, 0.5, 0.7, 1.0],
        labels = ["Low", "Medium", "High", "Critical"]
    )
    return model_df


# ── Save artefacts ────────────────────────────────────────────
def save_artefacts(lr, rf, scaler,
                   model_df: pd.DataFrame,
                   model_dir: str = MODEL_DIR,
                   data_processed: str = DATA_PROCESSED) -> None:
    """Save model pkl files and predictions CSV."""
    joblib.dump(rf,     os.path.join(model_dir, "churn_rf_model.pkl"))
    joblib.dump(lr,     os.path.join(model_dir, "churn_lr_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "churn_scaler.pkl"))

    print("\nModels saved to models/:")
    print("  churn_rf_model.pkl")
    print("  churn_lr_model.pkl")
    print("  churn_scaler.pkl")

    output = model_df[["user_id","churned","churn_risk_score","risk_tier"]].copy()
    out_path = os.path.join(data_processed, "churn_predictions.csv")
    output.to_csv(out_path, index=False)
    print(f"\nPredictions saved: churn_predictions.csv  ({len(output):,} riders)")


# ── Main pipeline ─────────────────────────────────────────────
def run_churn_model(data_raw: str       = DATA_RAW,
                    data_processed: str = DATA_PROCESSED,
                    model_dir: str      = MODEL_DIR) -> pd.DataFrame:
    """Full churn prediction pipeline."""
    print("=" * 55)
    print("RideWise — Churn Prediction Model")
    print("=" * 55)

    # 1. Load & engineer features
    model_df = load_and_engineer_features(data_raw)

    # 2. Train models
    lr, rf, scaler, X_train, X_test, y_train, y_test, lr_prob, rf_prob = \
        train_models(model_df)

    lr_pred = lr.predict(scaler.transform(X_test))
    rf_pred = rf.predict(X_test)

    # 3. Cross-validation
    cross_validate_models(lr, rf, scaler, X_train, y_train)

    # 4. Visualisations
    plot_roc_curves(y_test, lr_prob, rf_prob)
    plot_confusion_matrices(y_test, lr_pred, rf_pred)
    plot_feature_importance(rf)

    # 5. Risk scoring
    model_df = assign_risk_scores(model_df, rf)
    plot_risk_tiers(model_df)

    print("\nRisk tier distribution:")
    print(model_df["risk_tier"].value_counts().sort_index())

    # 6. Save everything
    save_artefacts(lr, rf, scaler, model_df, model_dir, data_processed)

    print("\nChurn prediction complete ✓")
    return model_df


if __name__ == "__main__":
    run_churn_model()
