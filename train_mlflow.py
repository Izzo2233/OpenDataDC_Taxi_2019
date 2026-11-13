import os
import time
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "OpenDataDC_Taxi_2019"
REGISTERED_MODEL_NAME = "dc-taxi-fare-model"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(tracking_uri=TRACKING_URI)


def load_data(x_path: str = "data/processed/X_clean.csv", y_path: str = "data/processed/y_target.csv"):
    print(f"Loading dataset from {x_path} and {y_path}...")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()
    X["ORIGINZIP"] = X["ORIGINZIP"].astype(str)
    X["is_coords_missing"] = X["ORIGIN_BLOCK_LATITUDE"].isna().astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    num_cols = [
        "ORIGIN_BLOCK_LATITUDE",
        "ORIGIN_BLOCK_LONGITUDE",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "is_coords_missing",
    ]
    low_card_cols = ["AIRPORT", "ORIGINSTATE", "TRIPTYPE"]
    high_card_cols = ["ORIGINCITY", "ORIGINZIP"]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_cols),
            ("target", TargetEncoder(smooth="auto", random_state=42), high_card_cols),
        ]
    )


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    candidates = {
        "Ridge": {
            "estimator": Ridge(alpha=10.0, random_state=42),
            "scale_numeric": True,
            "params": {"model_family": "Ridge", "alpha": 10.0},
        },
        "LightGBM": {
            "estimator": LGBMRegressor(
                n_estimators=300,
                learning_rate=0.08,
                num_leaves=63,
                subsample_for_bin=200000,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            ),
            "scale_numeric": False,
            "params": {
                "model_family": "LightGBM",
                "n_estimators": 300,
                "learning_rate": 0.08,
                "num_leaves": 63,
            },
        },
        "XGBoost": {
            "estimator": XGBRegressor(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=8,
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            ),
            "scale_numeric": False,
            "params": {
                "model_family": "XGBoost",
                "n_estimators": 300,
                "learning_rate": 0.08,
                "max_depth": 8,
                "tree_method": "hist",
            },
        },
    }

    input_example = X_train.iloc[:5].copy()
    best_rmse = float("inf")
    best_model_name = None
    best_version = None

    for name, cfg in candidates.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=cfg["scale_numeric"])),
                ("model", cfg["estimator"]),
            ]
        )

        with mlflow.start_run(run_name=f"{name}_baseline"):
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = round(time.time() - t0, 2)

            preds = pipeline.predict(X_test)
            rmse = float(root_mean_squared_error(y_test, preds))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            print(f"{name} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Time: {train_time}s")

            mlflow.log_params(cfg["params"])
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "train_seconds": train_time,
            })

            signature = infer_signature(input_example, pipeline.predict(input_example))
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=REGISTERED_MODEL_NAME, serialization_format="cloudpickle",
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_version = model_info.registered_model_version

    if best_version is not None:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=str(best_version),
        )
        print(
            f"\nPromoted {best_model_name} (Version {best_version}, RMSE={best_rmse:.4f}) "
            f"to '{REGISTERED_MODEL_NAME}@champion'"
        )


if __name__ == "__main__":
    main()
