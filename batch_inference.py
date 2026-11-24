import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_champion_artifacts(champion_dir: Path):
    meta_path = champion_dir / "champion_metadata.json"
    prep_path = champion_dir / "champion_preprocessor.joblib"
    model_path = champion_dir / "champion_model.joblib"

    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    preprocessor = joblib.load(prep_path) if prep_path.exists() else None
    model = joblib.load(model_path)
    return metadata, preprocessor, model


def get_expected_features(metadata: dict, preprocessor, model, df: pd.DataFrame) -> list[str]:
    for key in ("features", "feature_names", "input_features", "columns"):
        if isinstance(metadata.get(key), list) and metadata[key]:
            return metadata[key]
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    target = metadata.get("target", metadata.get("target_column"))
    return [c for c in df.columns if c != target]


def main():
    parser = argparse.ArgumentParser(description="Offline batch inference using Champion model")
    parser.add_argument("--champion-dir", type=str, default="models/champion", help="Directory with champion artifacts")
    parser.add_argument("--input", type=str, default="data/test_data.parquet", help="Input .parquet or .csv file")
    parser.add_argument("--output", type=str, default="data/batch_predictions.csv", help="Output .csv or .parquet file")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick test runs")
    args = parser.parse_args()

    champion_dir = Path(args.champion_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata, preprocessor, model = load_champion_artifacts(champion_dir)
    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    if args.limit:
        df = df.head(args.limit).copy()

    features = get_expected_features(metadata, preprocessor, model, df)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required columns in {input_path}: {missing[:10]}. "
            f"Available columns: {list(df.columns)}"
        )

    X = df[features]
    X_transformed = preprocessor.transform(X) if preprocessor is not None else X
    preds = model.predict(X_transformed)

    result_df = df.copy()
    result_df["prediction"] = preds

    if output_path.suffix == ".parquet":
        result_df.to_parquet(output_path, index=False)
    else:
        result_df.to_csv(output_path, index=False)

    model_name = metadata.get("model_name", metadata.get("name", "champion_model"))
    print(f"Champion model : {model_name}")
    print(f"Input file     : {input_path} ({len(result_df):,} rows)")
    print(f"Features used  : {len(features)}")
    print(f"Predictions    : min={np.min(preds):.2f}, mean={np.mean(preds):.2f}, max={np.max(preds):.2f}")
    print(f"Saved output   : {output_path}")

    target_col = metadata.get("target", metadata.get("target_column"))
    if not target_col:
        for candidate in ("trip_duration", "duration", "target", "y"):
            if candidate in result_df.columns:
                target_col = candidate
                break

    if target_col and target_col in result_df.columns:
        y_true = result_df[target_col].to_numpy()
        mae = float(np.mean(np.abs(y_true - preds)))
        rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
        print(f"Offline eval   : MAE={mae:.4f} | RMSE={rmse:.4f} (target='{target_col}')")

    print("\nSample predictions:")
    preview_cols = [c for c in features[:5] if c in result_df.columns]
    if target_col and target_col in result_df.columns:
        preview_cols.append(target_col)
    preview_cols.append("prediction")
    print(result_df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
