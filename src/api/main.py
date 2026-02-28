from fastapi import FastAPI

app = FastAPI(title="RideWise Churn API")

@app.get("/")
def home():
    return {"message": "RideWise API running 🚀"}
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="RideWise Churn API")

# -----------------------
# Request Model
# -----------------------
class ChurnRequest(BaseModel):
    total_trips: float = Field(ge=0)
    avg_fare: float = Field(ge=0)
    total_spent: float = Field(ge=0)
    recency_days: float = Field(ge=0)
    tenure_days: float = Field(ge=0)
    was_referred: int = Field(ge=0, le=1)

# -----------------------
# Response Model
# -----------------------
class ChurnResponse(BaseModel):
    churn_probability: float
    risk_band: str

# -----------------------
# Load Model
# -----------------------
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"

model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "RideWise API running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# -----------------------
# Predict Endpoint
# -----------------------
@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):

    X = np.array([[
        req.total_trips,
        req.avg_fare,
        req.total_spent,
        req.recency_days,
        req.tenure_days,
        req.was_referred
    ]])

    proba = float(model.predict_proba(X)[0, 1])

    if proba >= 0.7:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_probability": proba,
        "risk_band": risk
    }