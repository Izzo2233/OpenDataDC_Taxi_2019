import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/dc-taxi-fare-model@champion")

ml_models = {}


def load_champion_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    ml_models["champion"] = mlflow.pyfunc.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading champion model from {MODEL_URI} (Tracking URI: {TRACKING_URI})...")
    load_champion_model()
    print("Champion model loaded successfully.")
    yield
    ml_models.clear()


app = FastAPI(
    title="DC Taxi Fare Prediction API",
    version="2.2.0",
    lifespan=lifespan,
)

# Expose Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)


class TaxiTripRequest(BaseModel):
    ORIGIN_BLOCK_LATITUDE: Optional[float] = Field(38.8977, description="Origin latitude (nullable)")
    ORIGIN_BLOCK_LONGITUDE: Optional[float] = Field(-77.0365, description="Origin longitude (nullable)")
    hour: int = Field(14, ge=0, le=23)
    dayofweek: int = Field(2, ge=0, le=6)
    month: int = Field(6, ge=1, le=12)
    is_weekend: int = Field(0, ge=0, le=1)
    ORIGINCITY: str = Field("WASHINGTON")
    ORIGINSTATE: str = Field("DC")
    ORIGINZIP: str = Field("20001")
    AIRPORT: str = Field("N")
    TRIPTYPE: str = Field("Ordinal")


class PredictionResponse(BaseModel):
    predicted_fare: float
    model_uri: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "champion" in ml_models,
        "model_uri": MODEL_URI,
    }


@app.post("/reload-model")
def reload_model():
    try:
        load_champion_model()
        return {"status": "reloaded", "model_uri": MODEL_URI}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {exc}")


@app.post("/predict", response_model=PredictionResponse)
def predict_fare(trip: TaxiTripRequest):
    if "champion" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    df = pd.DataFrame([trip.model_dump()])
    df["ORIGINZIP"] = df["ORIGINZIP"].astype(str)
    df["is_coords_missing"] = df["ORIGIN_BLOCK_LATITUDE"].isna().astype(int)

    prediction = ml_models["champion"].predict(df)[0]
    return PredictionResponse(
        predicted_fare=round(float(prediction), 2),
        model_uri=MODEL_URI,
    )