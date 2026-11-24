import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

HUBS = {
    "dist_dca_km": (38.8512, -77.0402),       # Reagan National Airport (DCA)
    "dist_iad_km": (38.9531, -77.4565),       # Dulles International Airport (IAD)
    "dist_union_km": (38.8977, -77.0063),     # Union Station
    "dist_downtown_km": (38.9072, -77.0369),  # Downtown DC
}

NUMERICAL_FEATURES = [
    "ORIGIN_BLOCK_LATITUDE", "ORIGIN_BLOCK_LONGITUDE",
    "hour", "dayofweek", "month", "is_weekend",
    "dist_dca_km", "dist_iad_km", "dist_union_km", "dist_downtown_km",
    "is_rush_hour", "hour_sin", "hour_cos",
]

CATEGORICAL_FEATURES = ["ORIGINCITY", "ORIGINSTATE", "ORIGINZIP", "AIRPORT", "TRIPTYPE"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def patch_sklearn_compat(est):
    if isinstance(est, SimpleImputer) and not hasattr(est, "_fill_dtype"):
        est._fill_dtype = est.statistics_.dtype if hasattr(est, "statistics_") else np.dtype("O")
    if hasattr(est, "transformers_"):
        for _, trans, _ in est.transformers_:
            patch_sklearn_compat(trans)
    if hasattr(est, "steps"):
        for _, step in est.steps:
            patch_sklearn_compat(step)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "ORIGINDATETIME_TR" in out.columns and not {"hour", "dayofweek", "month", "is_weekend"}.issubset(out.columns):
        dt = pd.to_datetime(out["ORIGINDATETIME_TR"], errors="coerce")
        out["hour"] = dt.dt.hour
        out["dayofweek"] = dt.dt.dayofweek
        out["month"] = dt.dt.month
        out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    for col_name, (hub_lat, hub_lon) in HUBS.items():
        if col_name not in out.columns:
            out[col_name] = haversine_km(
                out["ORIGIN_BLOCK_LATITUDE"],
                out["ORIGIN_BLOCK_LONGITUDE"],
                hub_lat,
                hub_lon,
            )

    if "is_rush_hour" not in out.columns:
        out["is_rush_hour"] = (
            ((out["dayofweek"] < 5) & (out["hour"].between(7, 9) | out["hour"].between(16, 19)))
            .astype(int)
        )
    if "hour_sin" not in out.columns:
        out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    if "hour_cos" not in out.columns:
        out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)

    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(str)

    return out


def main():
    parser = argparse.ArgumentParser(description="Offline batch inference using Champion model")
    parser.add_argument("--champion-dir", type=str, default="models/champion", help="Champion artifacts directory")
    parser.add_argument("--input", type=str, default="data/test_data.parquet", help="Input .parquet or .csv file")
    parser.add_argument("--output", type=str, default="data/batch_predictions.parquet", help="Output .parquet or .csv")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick runs")
    args = parser.parse_args()

    champion_dir = Path(args.champion_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = champion_dir / "champion_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    preprocessor = joblib.load(champion_dir / "champion_preprocessor.joblib")
    patch_sklearn_compat(preprocessor)
    model = joblib.load(champion_dir / "champion_model.joblib")

    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    if args.limit:
        df = df.head(args.limit).copy()

    df_feat = engineer_features(df)
    missing = [c for c in ALL_FEATURES if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing required columns after feature engineering: {missing}")

    X_transformed = preprocessor.transform(df_feat[ALL_FEATURES])
    preds = model.predict(X_transformed)

    result_df = df.copy()
    result_df["predicted_duration"] = preds

    if output_path.suffix == ".parquet":
        result_df.to_parquet(output_path, index=False)
    else:
        result_df.to_csv(output_path, index=False)

    print(f"Champion model : {metadata.get('model_name', 'champion_model')}")
    print(f"Input file     : {input_path} ({len(result_df):,} rows)")
    print(f"Predictions    : min={np.min(preds):.2f}, mean={np.mean(preds):.2f}, max={np.max(preds):.2f}")
    print(f"Saved output   : {output_path}")

    if "DURATION" in result_df.columns:
        y_true = result_df["DURATION"].to_numpy()
        mae = float(np.mean(np.abs(y_true - preds)))
        rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
        print(f"Offline eval   : MAE={mae:.4f} | RMSE={rmse:.4f} (target='DURATION')")

    preview_cols = [c for c in ["ORIGINDATETIME_TR", "MILEAGE", "AIRPORT", "DURATION", "predicted_duration"] if c in result_df.columns]
    print("\nSample predictions:")
    print(result_df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
