
import os
import pandas as pd
import numpy as np

print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

# Load April, July and October 2019
data_path = os.path.join("data", "raw")
months = [4, 7, 10]

dfs = []

for month in months:
    file_path = os.path.join(
        data_path,
        f"taxi_2019_{month:02d}.csv"
    )

    print(f"Loading month {month}...")

    temp = pd.read_csv(file_path, low_memory=False)
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

print("Raw dataset shape:", df.shape)

# Convert pickup datetime
df["ORIGINDATETIME_TR"] = pd.to_datetime(
    df["ORIGINDATETIME_TR"],
    errors="coerce"
)

# Create target from DURATION (seconds)
df["DURATION"] = pd.to_numeric(
    df["DURATION"], errors="coerce"
)

df["trip_duration_minutes"] = df["DURATION"] / 60

# Keep valid target values: above 0 and up to 180 minutes
df_clean = df[
    df["trip_duration_minutes"].between(0, 180, inclusive="neither")
].copy()

# Feature engineering
df_clean["hour"] = df_clean["ORIGINDATETIME_TR"].dt.hour
df_clean["dayofweek"] = df_clean["ORIGINDATETIME_TR"].dt.dayofweek
df_clean["month"] = df_clean["ORIGINDATETIME_TR"].dt.month

df_clean["is_weekend"] = (
    df_clean["dayofweek"].isin([5, 6]).astype(int)
)

NUMERICAL_FEATURES = [
    "ORIGIN_BLOCK_LATITUDE",
    "ORIGIN_BLOCK_LONGITUDE",
    "hour",
    "dayofweek",
    "month",
    "is_weekend"
]

CATEGORICAL_FEATURES = [
    "ORIGINCITY",
    "ORIGINSTATE",
    "ORIGINZIP",
    "AIRPORT",
    "TRIPTYPE"
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "trip_duration_minutes"

# Check columns
missing_columns = [
    col for col in FEATURES + [TARGET]
    if col not in df_clean.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Prepare X and y
X = df_clean[FEATURES].copy()
y = df_clean[TARGET].copy()

X["ORIGINZIP"] = X["ORIGINZIP"].astype("string")

valid_target = y.notna() & np.isfinite(y) & (y > 0)

X = X.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)

# Save outputs
os.makedirs("data/processed", exist_ok=True)

X.to_csv("data/processed/X_features.csv", index=False)
y.to_csv("data/processed/y_target.csv", index=False)

print("\nFEATURE ENGINEERING COMPLETED")
print("=" * 60)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nMissing values:")
print(X.isna().sum())
print("\nTarget statistics:")
print(y.describe())

print("\nSaved:")
print("data/processed/X_features.csv")
print("data/processed/y_target.csv")
