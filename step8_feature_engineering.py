import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

print("STEP 8: DOMAIN FEATURE ENGINEERING & XGBOOST RE-EVALUATION")
print("=" * 65)

# 1. Load cleaned features and target
X = pd.read_csv("data/processed/X_clean.csv")
y = pd.read_csv("data/processed/y_target.csv").squeeze("columns")

# 2. Clean newly discovered ZIP placeholders ('00000', '-----', etc.)
X["ORIGINZIP"] = X["ORIGINZIP"].astype(str).str.strip()
extra_zip_missing = {"00000", "0", "-----", "---", "----", "nan", "None", "<NA>"}
X.loc[X["ORIGINZIP"].isin(extra_zip_missing), "ORIGINZIP"] = "Unknown"

# 3. Engineer Haversine Distances (in km) to major DC trip hubs
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

hubs = {
    "dist_dca_km": (38.8512, -77.0402),       # Reagan National Airport (DCA)
    "dist_iad_km": (38.9531, -77.4565),       # Dulles International Airport (IAD)
    "dist_union_km": (38.8977, -77.0063),     # Union Station
    "dist_downtown_km": (38.9072, -77.0369),  # Downtown DC (Metro Center / K St)
}

for col_name, (hub_lat, hub_lon) in hubs.items():
    X[col_name] = haversine_km(
        X["ORIGIN_BLOCK_LATITUDE"],
        X["ORIGIN_BLOCK_LONGITUDE"],
        hub_lat,
        hub_lon
    )

# 4. Engineer Rush Hour & Cyclical Time Features
X["is_rush_hour"] = (
    ((X["dayofweek"] < 5) & (X["hour"].between(7, 9) | X["hour"].between(16, 19)))
    .astype(int)
)
X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24.0)
X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24.0)

CATEGORICAL_FEATURES = ["ORIGINCITY", "ORIGINSTATE", "ORIGINZIP", "AIRPORT", "TRIPTYPE"]
NUMERICAL_FEATURES = [
    "ORIGIN_BLOCK_LATITUDE", "ORIGIN_BLOCK_LONGITUDE",
    "hour", "dayofweek", "month", "is_weekend",
    "dist_dca_km", "dist_iad_km", "dist_union_km", "dist_downtown_km",
    "is_rush_hour", "hour_sin", "hour_cos"
]

for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype(str)

# 5. Same 85/15 Split (random_state=42 ensures identical test rows!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

preprocessor_v2 = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUMERICAL_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ]), CATEGORICAL_FEATURES),
    ],
    sparse_threshold=1.0
)

print("Transforming engineered features...")
X_train_proc = preprocessor_v2.fit_transform(X_train).astype(np.float32)
X_test_proc = preprocessor_v2.transform(X_test).astype(np.float32)

# 6. Train Enhanced XGBoost
xgb_v2 = XGBRegressor(
    n_estimators=400,
    learning_rate=0.08,
    max_depth=9,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

print("Training Enhanced XGBoost (v2)...")
start_time = time.time()
xgb_v2.fit(X_train_proc, y_train.to_numpy(dtype=np.float32))
train_time = time.time() - start_time

y_pred = xgb_v2.predict(X_test_proc)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 65)
print("BEFORE vs. AFTER FEATURE ENGINEERING (Identical 313,402 Test Set)")
print("=" * 65)
print(f"Baseline XGBoost (v1) -> MAE: 6.216 min | RMSE: 9.167 min | R2: 0.2079")
print(f"Enhanced XGBoost (v2) -> MAE: {mae:.3f} min | RMSE: {rmse:.3f} min | R2: {r2:.4f} (Trained in {train_time:.1f}s)")

# Save v2 artifacts
os.makedirs("models/v2", exist_ok=True)
joblib.dump(preprocessor_v2, "models/v2/preprocessor_v2.joblib")
joblib.dump(xgb_v2, "models/v2/xgboost_v2.joblib")
print("\nSaved enhanced pipeline & model to 'models/v2/'")
