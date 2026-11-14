import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

print("STEP 9: RESIDUAL & ERROR ANALYSIS (MODEL V2)")
print("=" * 65)

# 1. Reload dataset and recreate the exact 85/15 test split
X = pd.read_csv("data/processed/X_clean.csv")
y = pd.read_csv("data/processed/y_target.csv").squeeze("columns")

# Apply the same Step 8 cleaning & feature engineering
X["ORIGINZIP"] = X["ORIGINZIP"].astype(str).str.strip()
extra_zip_missing = {"00000", "0", "-----", "---", "----", "nan", "None", "<NA>"}
X.loc[X["ORIGINZIP"].isin(extra_zip_missing), "ORIGINZIP"] = "Unknown"

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

hubs = {
    "dist_dca_km": (38.8512, -77.0402),
    "dist_iad_km": (38.9531, -77.4565),
    "dist_union_km": (38.8977, -77.0063),
    "dist_downtown_km": (38.9072, -77.0369),
}
for col_name, (hub_lat, hub_lon) in hubs.items():
    X[col_name] = haversine_km(X["ORIGIN_BLOCK_LATITUDE"], X["ORIGIN_BLOCK_LONGITUDE"], hub_lat, hub_lon)

X["is_rush_hour"] = (((X["dayofweek"] < 5) & (X["hour"].between(7, 9) | X["hour"].between(16, 19))).astype(int))
X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24.0)
X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24.0)

for col in ["ORIGINCITY", "ORIGINSTATE", "ORIGINZIP", "AIRPORT", "TRIPTYPE"]:
    X[col] = X[col].astype(str)

_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 2. Load v2 preprocessor and model, generate predictions
preprocessor_v2 = joblib.load("models/v2/preprocessor_v2.joblib")
xgb_v2 = joblib.load("models/v2/xgboost_v2.joblib")

X_test_proc = preprocessor_v2.transform(X_test).astype(np.float32)
y_pred = xgb_v2.predict(X_test_proc)

# 3. Build Error Analysis DataFrame
eval_df = X_test.copy()
eval_df["actual_min"] = y_test.values
eval_df["pred_min"] = y_pred
eval_df["error_min"] = eval_df["pred_min"] - eval_df["actual_min"]   # + means overestimate, - means underestimate
eval_df["abs_error_min"] = eval_df["error_min"].abs()

# A. Error by Actual Trip Duration Bracket
bins = [0, 5, 10, 15, 25, 45, 1000]
labels = ["0-5 min", "5-10 min", "10-15 min", "15-25 min", "25-45 min", "45+ min"]
eval_df["duration_bucket"] = pd.cut(eval_df["actual_min"], bins=bins, labels=labels, right=False)

bucket_summary = eval_df.groupby("duration_bucket", observed=False).agg(
    Trip_Count=("actual_min", "count"),
    Share_Pct=("actual_min", lambda s: round(len(s) / len(eval_df) * 100, 1)),
    Avg_Actual=("actual_min", lambda s: round(s.mean(), 2)),
    Avg_Pred=("pred_min", lambda s: round(s.mean(), 2)),
    MAE=("abs_error_min", lambda s: round(s.mean(), 2)),
    Bias_Error=("error_min", lambda s: round(s.mean(), 2))
).reset_index()

print("\n1. ERROR BREAKDOWN BY ACTUAL TRIP DURATION:")
print("-" * 75)
print(bucket_summary.to_string(index=False))

# B. Error by Airport Status
airport_summary = eval_df.groupby("AIRPORT").agg(
    Trip_Count=("actual_min", "count"),
    Avg_Actual=("actual_min", lambda s: round(s.mean(), 2)),
    Avg_Pred=("pred_min", lambda s: round(s.mean(), 2)),
    MAE=("abs_error_min", lambda s: round(s.mean(), 2)),
    RMSE=("error_min", lambda s: round(np.sqrt(np.mean(s ** 2)), 2))
).reset_index()

print("\n2. ERROR BREAKDOWN BY AIRPORT FLAG:")
print("-" * 75)
print(airport_summary.to_string(index=False))

# C. Error by Rush Hour vs. Off-Peak
rush_summary = eval_df.groupby("is_rush_hour").agg(
    Trip_Count=("actual_min", "count"),
    Avg_Actual=("actual_min", lambda s: round(s.mean(), 2)),
    MAE=("abs_error_min", lambda s: round(s.mean(), 2)),
    RMSE=("error_min", lambda s: round(np.sqrt(np.mean(s ** 2)), 2))
).reset_index()
rush_summary["is_rush_hour"] = rush_summary["is_rush_hour"].map({0: "Off-Peak", 1: "Rush Hour"})

print("\n3. ERROR BREAKDOWN BY RUSH HOUR:")
print("-" * 75)
print(rush_summary.to_string(index=False))

# D. Top 10 Largest Errors (Outlier Inspection)
worst_10 = eval_df.sort_values(by="abs_error_min", ascending=False).head(10)[
    ["ORIGINCITY", "ORIGINSTATE", "ORIGINZIP", "AIRPORT", "hour", "actual_min", "pred_min", "abs_error_min"]
]
print("\n4. TOP 10 WORST PREDICTIONS (OUTLIER INSPECTION):")
print("-" * 75)
print(worst_10.to_string(index=False))

os.makedirs("models/v2", exist_ok=True)
bucket_summary.to_csv("models/v2/error_by_duration.csv", index=False)
print("\nSaved residual summary to 'models/v2/error_by_duration.csv'")
