import glob
import math
import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/dc-taxi-fare-model@champion")

ml_models = {}

PREDICTED_FARE_HISTOGRAM = Histogram(
    "dc_taxi_predicted_fare_dollars",
    "Distribution of predicted taxi fares in USD",
    buckets=[5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0, 150.0],
)

LANDMARKS = {
    "dist_dca_km": (38.8512, -77.0402),
    "dist_iad_km": (38.9531, -77.4565),
    "dist_union_km": (38.8977, -77.0063),
    "dist_downtown_km": (38.9072, -77.0369),
}


def patch_sklearn_compat(est):
    if isinstance(est, SimpleImputer) and not hasattr(est, "_fill_dtype"):
        est._fill_dtype = est.statistics_.dtype if hasattr(est, "statistics_") else np.dtype("O")
    if hasattr(est, "transformers_"):
        for _, trans, _ in est.transformers_:
            patch_sklearn_compat(trans)
    if hasattr(est, "steps"):
        for _, step in est.steps:
            patch_sklearn_compat(step)
    return est


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ORIGINZIP"] = df["ORIGINZIP"].astype(str)
    df["is_coords_missing"] = df["ORIGIN_BLOCK_LATITUDE"].isna().astype(int)

    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["is_rush_hour"] = (
        (df["is_weekend"] == 0)
        & (df["hour"].between(7, 9) | df["hour"].between(16, 19))
    ).astype(int)

    lat = df["ORIGIN_BLOCK_LATITUDE"].fillna(38.9072).iloc[0]
    lon = df["ORIGIN_BLOCK_LONGITUDE"].fillna(-77.0369).iloc[0]
    for col_name, (target_lat, target_lon) in LANDMARKS.items():
        df[col_name] = haversine_km(float(lat), float(lon), target_lat, target_lon)

    return df


def align_transformed_matrix(X_in, preprocessor, model):
    if hasattr(X_in, "toarray"):
        X_in = X_in.toarray()
    X_arr = np.asarray(X_in)

    if not hasattr(model, "get_booster"):
        return X_arr

    booster = model.get_booster()
    expected_n = booster.num_features()
    b_names = booster.feature_names

    if b_names and hasattr(preprocessor, "get_feature_names_out"):
        p_names = list(preprocessor.get_feature_names_out())
        if len(p_names) == X_arr.shape[1]:
            df_trans = pd.DataFrame(X_arr, columns=p_names)
            for col in b_names:
                if col not in df_trans.columns:
                    df_trans[col] = 0.0
            return df_trans[b_names].to_numpy()

    if expected_n and X_arr.shape[1] != expected_n:
        if X_arr.shape[1] > expected_n:
            return X_arr[:, :expected_n]
        padded = np.zeros((X_arr.shape[0], expected_n), dtype=X_arr.dtype)
        padded[:, : X_arr.shape[1]] = X_arr
        return padded

    return X_arr


def load_champion_model():
    err_msg = ""
    ml_models.pop("preprocessor", None)

    # 1. Try MLflow Registry first
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        ml_models["champion"] = mlflow.pyfunc.load_model(MODEL_URI)
        ml_models["active_uri"] = MODEL_URI
        ml_models["load_error"] = None
        print(f"Loaded champion model from MLflow Registry: {MODEL_URI}")
        return
    except Exception as exc:
        err_msg = str(exc)

    # 2. Try local MLflow artifact directories (models/champion, models/v2, or mlruns_artifacts/**/MLmodel)
    local_mlmodel_dirs = []
    for candidate_dir in ["models/champion", "models/v2"]:
        if os.path.exists(os.path.join(candidate_dir, "MLmodel")):
            local_mlmodel_dirs.append(candidate_dir)
    for mlmodel_path in sorted(glob.glob("models/**/MLmodel", recursive=True) + glob.glob("mlruns_artifacts/**/MLmodel", recursive=True)):
        d = os.path.dirname(mlmodel_path)
        if d not in local_mlmodel_dirs:
            local_mlmodel_dirs.append(d)

    for artifact_dir in local_mlmodel_dirs:
        try:
            ml_models["champion"] = mlflow.pyfunc.load_model(artifact_dir)
            ml_models["active_uri"] = f"mlflow-local:{artifact_dir}"
            ml_models["load_error"] = None
            print(f"Loaded local MLflow champion bundle: {artifact_dir}")
            return
        except Exception as local_err:
            err_msg = f"{err_msg} | {artifact_dir}: {local_err}"

    # 3. Fallback: load standalone preprocessor + XGBRegressor from models/
    all_joblibs = sorted(glob.glob("models/**/*.joblib", recursive=True) + glob.glob("models/**/*.pkl", recursive=True))
    regressor = None
    regressor_path = None

    for path in all_joblibs:
        obj = patch_sklearn_compat(joblib.load(path))
        if hasattr(obj, "transform") and not hasattr(obj, "predict"):
            ml_models["preprocessor"] = obj
            print(f"Loaded preprocessor: {path}")
        elif hasattr(obj, "predict") and "xgboost" in os.path.basename(path).lower():
            regressor = obj
            regressor_path = path

    if regressor is None:
        for path in all_joblibs:
            obj = patch_sklearn_compat(joblib.load(path))
            if hasattr(obj, "predict"):
                regressor = obj
                regressor_path = path
                break

    if regressor is not None:
        ml_models["champion"] = regressor
        ml_models["active_uri"] = f"local:{regressor_path}"
        ml_models["load_error"] = None
        print(f"Loaded fallback regressor: {regressor_path}")
    else:
        ml_models["load_error"] = f"Registry unavailable ({err_msg}) and no regressor found in ./models"
        print(f"WARNING: {ml_models['load_error']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_champion_model()
    yield
    ml_models.clear()


app = FastAPI(
    title="DC Taxi Fare Prediction API",
    version="3.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


class TaxiTripRequest(BaseModel):
    ORIGIN_BLOCK_LATITUDE: Optional[float] = Field(38.8977)
    ORIGIN_BLOCK_LONGITUDE: Optional[float] = Field(-77.0365)
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
        "status": "healthy" if "champion" in ml_models else "degraded",
        "model_loaded": "champion" in ml_models,
        "preprocessor_loaded": "preprocessor" in ml_models,
        "model_uri": ml_models.get("active_uri", MODEL_URI),
        "error": ml_models.get("load_error"),
    }


@app.post("/reload-model")
def reload_model():
    load_champion_model()
    if "champion" not in ml_models:
        raise HTTPException(status_code=500, detail=ml_models.get("load_error", "Model load failed"))
    return {"status": "reloaded", "model_uri": ml_models.get("active_uri", MODEL_URI)}


@app.post("/predict", response_model=PredictionResponse)
def predict_fare(trip: TaxiTripRequest):
    if "champion" not in ml_models:
        raise HTTPException(status_code=503, detail=ml_models.get("load_error", "Model not loaded yet"))

    try:
        raw_df = pd.DataFrame([trip.model_dump()])
        feat_df = engineer_features(raw_df)
        if "preprocessor" in ml_models:
            X_trans = ml_models["preprocessor"].transform(feat_df)
            X_in = align_transformed_matrix(X_trans, ml_models["preprocessor"], ml_models["champion"])
        else:
            X_in = feat_df

        fare = round(float(ml_models["champion"].predict(X_in)[0]), 2)
        PREDICTED_FARE_HISTOGRAM.observe(fare)

        return PredictionResponse(
            predicted_fare=fare,
            model_uri=ml_models.get("active_uri", MODEL_URI),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction input mismatch: {exc}")