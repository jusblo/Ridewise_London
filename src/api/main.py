"""
RideWise — FastAPI Application
================================
Endpoints:
  GET  /health               — system status & uptime
  POST /predict/churn        — churn probability for a rider
  POST /customer/segment     — segment assignment for a rider
  GET  /dashboard/metrics    — key business KPIs

Usage:
    uvicorn src.api.main:app --reload --port 8000

Then open: http://127.0.0.1:8000/docs
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR      = os.path.join(BASE_DIR, "models")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

# ── App startup time (for uptime calculation) ─────────────────
START_TIME = time.time()

# ── Feature columns (must match training order exactly) ───────
FEATURE_COLS = [
    "age", "account_age_days", "was_referred",
    "loyalty_encoded", "city_encoded",
    "n_trips", "avg_fare", "avg_surge", "total_spend", "tip_rate",
    "peak_hour_rate", "weekend_ratio", "days_since_last",
    "trips_last_7d", "trips_last_30d", "trips_last_60d", "trips_last_90d",
    "activity_trend_30d",
    "n_sessions", "session_last_30d", "avg_time_on_app",
    "conv_rate", "days_since_session",
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
    "rfm_combined_score", "engagement_score",
]

LOYALTY_MAP = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
CITY_MAP    = {"Cairo": 0, "Lagos": 1, "Nairobi": 2}

CLUSTER_COLS = [
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
    "avg_fare", "avg_surge", "tip_rate", "peak_hour_rate", "weekend_ratio",
    "session_last_30d", "avg_time_on_app", "session_conversion_rate",
    "engagement_score", "account_age_days", "activity_trend_30d",
]


# ── Load models ───────────────────────────────────────────────
def load_models():
    """Load all saved model artefacts."""
    models = {}
    try:
        models["rf"]            = joblib.load(os.path.join(MODEL_DIR, "churn_rf_model.pkl"))
        models["lr"]            = joblib.load(os.path.join(MODEL_DIR, "churn_lr_model.pkl"))
        models["churn_scaler"]  = joblib.load(os.path.join(MODEL_DIR, "churn_scaler.pkl"))
        models["kmeans"]        = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
        models["seg_scaler"]    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        print("All models loaded ✓")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    return models

MODELS = load_models()


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title       = "RideWise Customer Analytics API",
    description = "Churn prediction, customer segmentation and business metrics.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response schemas ────────────────────────────────
class RiderFeatures(BaseModel):
    """Input features for a single rider."""
    user_id:              str   = Field(...,  example="R00001")
    age:                  float = Field(28.0, example=28.0)
    account_age_days:     int   = Field(180,  example=180)
    was_referred:         int   = Field(0,    example=0,    ge=0, le=1)
    loyalty_status:       str   = Field("Bronze", example="Bronze")
    city:                 str   = Field("Lagos",  example="Lagos")
    n_trips:              int   = Field(12,   example=12)
    avg_fare:             float = Field(15.0, example=15.0)
    avg_surge:            float = Field(1.1,  example=1.1)
    total_spend:          float = Field(180.0,example=180.0)
    tip_rate:             float = Field(0.4,  example=0.4)
    peak_hour_rate:       float = Field(0.3,  example=0.3)
    weekend_ratio:        float = Field(0.25, example=0.25)
    days_since_last:      int   = Field(5,    example=5)
    trips_last_7d:        int   = Field(2,    example=2)
    trips_last_30d:       int   = Field(6,    example=6)
    trips_last_60d:       int   = Field(10,   example=10)
    trips_last_90d:       int   = Field(12,   example=12)
    activity_trend_30d:   float = Field(0.6,  example=0.6)
    n_sessions:           int   = Field(20,   example=20)
    session_last_30d:     int   = Field(5,    example=5)
    avg_time_on_app:      float = Field(400.0,example=400.0)
    conv_rate:            float = Field(0.5,  example=0.5)
    days_since_session:   int   = Field(2,    example=2)
    rfm_recency_score:    int   = Field(4,    example=4,    ge=1, le=5)
    rfm_frequency_score:  int   = Field(3,    example=3,    ge=1, le=5)
    rfm_monetary_score:   int   = Field(3,    example=3,    ge=1, le=5)
    rfm_combined_score:   float = Field(3.33, example=3.33)
    engagement_score:     float = Field(3.35, example=3.35)
    session_conversion_rate: float = Field(0.5, example=0.5)


class ChurnResponse(BaseModel):
    user_id:          str
    churn_probability: float
    churn_prediction: int
    risk_tier:        str
    model_used:       str


class SegmentResponse(BaseModel):
    user_id:      str
    cluster_id:   int
    segment_name: str
    description:  str


class HealthResponse(BaseModel):
    status:        str
    uptime_seconds: float
    models_loaded: list
    version:       str


class MetricsResponse(BaseModel):
    total_riders:       Optional[int]
    churn_rate_pct:     Optional[float]
    high_risk_riders:   Optional[int]
    critical_risk_riders: Optional[int]
    avg_churn_score:    Optional[float]
    segment_distribution: Optional[dict]


# ── Helper functions ──────────────────────────────────────────
def rider_to_array(rider: RiderFeatures) -> np.ndarray:
    """Convert a RiderFeatures request into a numpy array for prediction."""
    loyalty_enc = LOYALTY_MAP.get(rider.loyalty_status, 0)
    city_enc    = CITY_MAP.get(rider.city, 1)

    values = [
        rider.age, rider.account_age_days, rider.was_referred,
        loyalty_enc, city_enc,
        rider.n_trips, rider.avg_fare, rider.avg_surge,
        rider.total_spend, rider.tip_rate, rider.peak_hour_rate,
        rider.weekend_ratio, rider.days_since_last,
        rider.trips_last_7d, rider.trips_last_30d,
        rider.trips_last_60d, rider.trips_last_90d,
        rider.activity_trend_30d,
        rider.n_sessions, rider.session_last_30d,
        rider.avg_time_on_app, rider.conv_rate, rider.days_since_session,
        rider.rfm_recency_score, rider.rfm_frequency_score,
        rider.rfm_monetary_score, rider.rfm_combined_score,
        rider.engagement_score,
    ]
    return np.array(values).reshape(1, -1)


def get_risk_tier(prob: float) -> str:
    """Convert a probability to a named risk tier."""
    if prob < 0.3:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.7:
        return "High"
    else:
        return "Critical"


SEGMENT_NAMES = {
    0: ("Champions",    "High RFM scores. Most valuable, loyal riders."),
    1: ("Loyal Riders", "Good frequency and spend. Strong retention potential."),
    2: ("At-Risk",      "Declining activity. Need re-engagement campaigns."),
    3: ("Dormant",      "Low RFM scores. Highest churn risk."),
}


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check API health, uptime, and which models are loaded."""
    loaded = [k for k in MODELS if MODELS[k] is not None]
    return HealthResponse(
        status         = "healthy",
        uptime_seconds = round(time.time() - START_TIME, 2),
        models_loaded  = loaded,
        version        = "1.0.0",
    )


@app.post("/predict/churn", response_model=ChurnResponse, tags=["Prediction"])
def predict_churn(rider: RiderFeatures):
    """
    Predict the churn probability for a single rider.

    Returns:
    - **churn_probability**: float between 0 and 1
    - **churn_prediction**: 1 = likely to churn, 0 = likely to stay
    - **risk_tier**: Low / Medium / High / Critical
    - **model_used**: which model made the prediction
    """
    if "rf" not in MODELS:
        raise HTTPException(status_code=503,
                            detail="Churn model not loaded. Run the training pipeline first.")

    X = rider_to_array(rider)

    rf_prob = float(MODELS["rf"].predict_proba(X)[0, 1])
    prediction = int(rf_prob >= 0.5)
    risk_tier  = get_risk_tier(rf_prob)

    return ChurnResponse(
        user_id           = rider.user_id,
        churn_probability = round(rf_prob, 4),
        churn_prediction  = prediction,
        risk_tier         = risk_tier,
        model_used        = "Random Forest",
    )


@app.post("/customer/segment", response_model=SegmentResponse, tags=["Segmentation"])
def customer_segment(rider: RiderFeatures):
    """
    Assign a rider to their predicted customer segment.

    Segments:
    - **Champions**: High value, loyal riders
    - **Loyal Riders**: Good frequency and spend
    - **At-Risk**: Declining activity
    - **Dormant**: Low RFM, highest churn risk
    """
    if "kmeans" not in MODELS:
        raise HTTPException(status_code=503,
                            detail="Segmentation model not loaded. Run the training pipeline first.")

    cluster_values = [
        rider.rfm_recency_score, rider.rfm_frequency_score,
        rider.rfm_monetary_score, rider.avg_fare, rider.avg_surge,
        rider.tip_rate, rider.peak_hour_rate, rider.weekend_ratio,
        rider.session_last_30d, rider.avg_time_on_app,
        rider.session_conversion_rate, rider.engagement_score,
        rider.account_age_days, rider.activity_trend_30d,
    ]

    X_seg       = np.array(cluster_values).reshape(1, -1)
    X_scaled    = MODELS["seg_scaler"].transform(X_seg)
    cluster_id  = int(MODELS["kmeans"].predict(X_scaled)[0])
    name, desc  = SEGMENT_NAMES.get(cluster_id, ("Unknown", ""))

    return SegmentResponse(
        user_id      = rider.user_id,
        cluster_id   = cluster_id,
        segment_name = name,
        description  = desc,
    )


@app.get("/dashboard/metrics", response_model=MetricsResponse, tags=["Dashboard"])
def dashboard_metrics():
    """
    Aggregate business KPIs from the saved predictions file.

    Returns overall churn rate, risk tier counts,
    average churn score, and segment distribution.
    """
    pred_path = os.path.join(DATA_PROCESSED, "churn_predictions.csv")
    seg_path  = os.path.join(DATA_PROCESSED, "features_segmented.csv")

    if not os.path.exists(pred_path):
        raise HTTPException(
            status_code=503,
            detail="Predictions file not found. Run the churn model pipeline first."
        )

    preds = pd.read_csv(pred_path)

    total_riders       = len(preds)
    churn_rate_pct     = round(float(preds["churned"].mean()) * 100, 2)
    high_risk_riders   = int((preds["risk_tier"] == "High").sum())
    critical_riders    = int((preds["risk_tier"] == "Critical").sum())
    avg_churn_score    = round(float(preds["churn_risk_score"].mean()), 4)

    seg_dist = None
    if os.path.exists(seg_path):
        segs     = pd.read_csv(seg_path)
        seg_dist = segs["segment"].value_counts().to_dict() if "segment" in segs.columns else None

    return MetricsResponse(
        total_riders          = total_riders,
        churn_rate_pct        = churn_rate_pct,
        high_risk_riders      = high_risk_riders,
        critical_risk_riders  = critical_riders,
        avg_churn_score       = avg_churn_score,
        segment_distribution  = seg_dist,
    )
