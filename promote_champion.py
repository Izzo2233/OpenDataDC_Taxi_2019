import os
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import sparse
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

print("AUTO-ALIASING & PROMOTING CHAMPION MODEL")
print("=" * 75)

alias_dir = "models/aliases"
champ_dir = "models/champion"
os.makedirs(alias_dir, exist_ok=True)
os.makedirs(champ_dir, exist_ok=True)

# 1. Auto-copy existing trained models & preprocessors into models/aliases/
source_map = {
    "ridge": ("models/ridge_regression.joblib", "v1"),
    "sgd": ("models/sgd_regressor.joblib", "v1"),
    "rf": ("models/random_forest.joblib", "v1"),
    "xgb_v1": ("models/xgboost_regressor.joblib", "v1"),
    "xgb_v2": ("models/v2/xgboost_v2.joblib", "v2"),
}

for alias, (src, _) in source_map.items():
    if os.path.exists(src):
        shutil.copy2(src, f"{alias_dir}/{alias}.joblib")

if os.path.exists("data/prepared/preprocessor.joblib"):
    shutil.copy2("data/prepared/preprocessor.joblib", f"{alias_dir}/prep_v1.joblib")
if os.path.exists("models/v2/preprocessor_v2.joblib"):
    shutil.copy2("models/v2/preprocessor_v2.joblib", f"{alias_dir}/prep_v2.joblib")

# 2. Load Test Data
X_test_v1 = sparse.load_npz("data/prepared/X_test_processed.npz").astype(np.float32)
y_test = np.load("data/prepared/y_test.npy").astype(np.float32)

candidates = []
v1_preds = {}

# 3. Evaluate all v1 aliases
for alias in ["ridge", "sgd", "rf", "xgb_v1"]:
    path = f"{alias_dir}/{alias}.joblib"
    if os.path.exists(path):
        print(f"Evaluating '{alias}'...")
        model = joblib.load(path)
        preds = model.predict(X_test_v1)
        v1_preds[alias] = preds
        candidates.append({
            "alias": alias,
            "version": "v1",
            "model_path": path,
            "prep_path": f"{alias_dir}/prep_v1.joblib",
            "MAE": float(mean_absolute_error(y_test, preds)),
            "RMSE": float(root_mean_squared_error(y_test, preds)),
            "R2": float(r2_score(y_test, preds))
        })

# 4. Evaluate Weighted Ensemble if v1 models exist
if all(k in v1_preds for k in ["xgb_v1", "rf", "ridge"]):
    print("Evaluating 'ensemble' (65% XGB_v1 + 25% RF + 10% Ridge)...")
    weights = {"xgb_v1": 0.65, "rf": 0.25, "ridge": 0.10}
    preds_ens = (
        weights["xgb_v1"] * v1_preds["xgb_v1"]
        + weights["rf"] * v1_preds["rf"]
        + weights["ridge"] * v1_preds["ridge"]
    )
    ens_path = f"{alias_dir}/ensemble.joblib"
    joblib.dump({
        "type": "weighted_blend",
        "weights": weights,
        "model_paths": {k: f"{alias_dir}/{k}.joblib" for k in weights}
    }, ens_path)

    candidates.append({
        "alias": "ensemble",
        "version": "v1",
        "model_path": ens_path,
        "prep_path": f"{alias_dir}/prep_v1.joblib",
        "MAE": float(mean_absolute_error(y_test, preds_ens)),
        "RMSE": float(root_mean_squared_error(y_test, preds_ens)),
        "R2": float(r2_score(y_test, preds_ens))
    })

# 5. Evaluate xgb_v2 if Step 8 was run
xgb_v2_path = f"{alias_dir}/xgb_v2.joblib"
prep_v2_path = f"{alias_dir}/prep_v2.joblib"
if os.path.exists(xgb_v2_path) and os.path.exists(prep_v2_path):
    print("Evaluating 'xgb_v2'...")
    from sklearn.model_selection import train_test_split
    X_raw = pd.read_csv("data/processed/X_clean.csv")
    y_raw = pd.read_csv("data/processed/y_target.csv").squeeze("columns")

    X_raw["ORIGINZIP"] = X_raw["ORIGINZIP"].astype(str).str.strip()
    extra_missing = {"00000", "0", "-----", "---", "----", "nan", "None", "<NA>"}
    X_raw.loc[X_raw["ORIGINZIP"].isin(extra_missing), "ORIGINZIP"] = "Unknown"

    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))

    for col_name, (h_lat, h_lon) in {
        "dist_dca_km": (38.8512, -77.0402),
        "dist_iad_km": (38.9531, -77.4565),
        "dist_union_km": (38.8977, -77.0063),
        "dist_downtown_km": (38.9072, -77.0369),
    }.items():
        X_raw[col_name] = haversine_km(X_raw["ORIGIN_BLOCK_LATITUDE"], X_raw["ORIGIN_BLOCK_LONGITUDE"], h_lat, h_lon)

    X_raw["is_rush_hour"] = (((X_raw["dayofweek"] < 5) & (X_raw["hour"].between(7, 9) | X_raw["hour"].between(16, 19))).astype(int))
    X_raw["hour_sin"] = np.sin(2 * np.pi * X_raw["hour"] / 24.0)
    X_raw["hour_cos"] = np.cos(2 * np.pi * X_raw["hour"] / 24.0)
    for col in ["ORIGINCITY", "ORIGINSTATE", "ORIGINZIP", "AIRPORT", "TRIPTYPE"]:
        X_raw[col] = X_raw[col].astype(str)

    _, X_test_raw, _, y_test_v2 = train_test_split(X_raw, y_raw, test_size=0.15, random_state=42)
    prep_v2 = joblib.load(prep_v2_path)
    xgb_v2 = joblib.load(xgb_v2_path)
    X_test_v2 = prep_v2.transform(X_test_raw).astype(np.float32)
    preds_v2 = xgb_v2.predict(X_test_v2)

    candidates.append({
        "alias": "xgb_v2",
        "version": "v2",
        "model_path": xgb_v2_path,
        "prep_path": prep_v2_path,
        "MAE": float(mean_absolute_error(y_test_v2, preds_v2)),
        "RMSE": float(root_mean_squared_error(y_test_v2, preds_v2)),
        "R2": float(r2_score(y_test_v2, preds_v2))
    })

if not candidates:
    raise FileNotFoundError("No trained .joblib models found in 'models/' or 'models/v2/'. Please verify your models folder.")

# 6. Rank Candidates & Promote Champion
df_rank = pd.DataFrame(candidates).sort_values(by="RMSE").reset_index(drop=True)
df_rank.index = df_rank.index + 1

print("\nALL CANDIDATES RANKED BY RMSE:")
print("-" * 75)
print(df_rank[["alias", "version", "MAE", "RMSE", "R2"]].round(4).to_string())

champion = df_rank.iloc[0]
champ_alias = champion["alias"]

shutil.copy2(champion["model_path"], f"{champ_dir}/champion_model.joblib")
if os.path.exists(champion["prep_path"]):
    shutil.copy2(champion["prep_path"], f"{champ_dir}/champion_preprocessor.joblib")

metadata = {
    "champion_alias": champ_alias,
    "feature_pipeline_version": champion["version"],
    "promoted_at": datetime.now().isoformat(timespec="seconds"),
    "test_set_size": int(len(y_test)),
    "metrics": {
        "MAE_min": round(float(champion["MAE"]), 4),
        "RMSE_min": round(float(champion["RMSE"]), 4),
        "R2_score": round(float(champion["R2"]), 4)
    },
    "artifacts": {
        "model": f"{champ_dir}/champion_model.joblib",
        "preprocessor": f"{champ_dir}/champion_preprocessor.joblib"
    }
}

with open(f"{champ_dir}/champion_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# Save full registry as well
df_rank[["alias", "version", "MAE", "RMSE", "R2"]].to_csv(f"{alias_dir}/final_comparison_table.csv", index=False)

print("\n" + "=" * 75)
print(f"CHAMPION SELECTED: '{champ_alias.upper()}' (RMSE: {champion['RMSE']:.3f} min | R2: {champion['R2']:.4f})")
print("=" * 75)
print(f"Saved production bundle to '{champ_dir}/':")
print("- champion_model.joblib")
print("- champion_preprocessor.joblib")
print("- champion_metadata.json")
