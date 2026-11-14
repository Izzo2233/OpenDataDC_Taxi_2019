
import pandas as pd
import numpy as np
import os

print("STEP 4: CLEANING FEATURE VALUES")
print("=" * 60)

# Load previously saved features
X = pd.read_csv("data/processed/X_features.csv")
y = pd.read_csv("data/processed/y_target.csv").squeeze("columns")

X_clean = X.copy()

# 1. Clean categorical text
CATEGORICAL_FEATURES = [
    "ORIGINCITY",
    "ORIGINSTATE",
    "ORIGINZIP",
    "AIRPORT",
    "TRIPTYPE"
]

for col in CATEGORICAL_FEATURES:
    X_clean[col] = (
        X_clean[col]
        .astype("string")
        .str.strip()
    )

# 2. Standardize missing and invalid text values
missing_labels = ["", "-", "--", "UNKNOWN", "NAN", "NONE"]

for col in CATEGORICAL_FEATURES:
    X_clean[col] = X_clean[col].replace(
        missing_labels,
        pd.NA,
        regex=False
    )

# 3. Standardize city and state capitalization
X_clean["ORIGINCITY"] = X_clean["ORIGINCITY"].str.title()
X_clean["ORIGINSTATE"] = X_clean["ORIGINSTATE"].str.upper()

# 4. Standardize Trip Type
X_clean["TRIPTYPE"] = X_clean["TRIPTYPE"].replace({
    "Transport DC": "TransportDC"
})

# 5. Treat missing AIRPORT as Unknown
X_clean["AIRPORT"] = X_clean["AIRPORT"].fillna("Unknown")

# 6. Fill missing categorical values
for col in [
    "ORIGINCITY",
    "ORIGINSTATE",
    "ORIGINZIP",
    "TRIPTYPE"
]:
    X_clean[col] = X_clean[col].fillna("Unknown")

# 7. Convert numerical features to numeric
NUMERICAL_FEATURES = [
    "ORIGIN_BLOCK_LATITUDE",
    "ORIGIN_BLOCK_LONGITUDE",
    "hour",
    "dayofweek",
    "month",
    "is_weekend"
]

for col in NUMERICAL_FEATURES:
    X_clean[col] = pd.to_numeric(
        X_clean[col],
        errors="coerce"
    )

# 8. Save cleaned features
os.makedirs("data/processed", exist_ok=True)

X_clean.to_csv(
    "data/processed/X_clean.csv",
    index=False
)

y.to_csv(
    "data/processed/y_target.csv",
    index=False
)

# 9. Report results
print("\nCLEANING COMPLETED")
print("=" * 60)

print("X shape:", X_clean.shape)
print("y shape:", y.shape)

print("\nCategorical unique values:")
for col in CATEGORICAL_FEATURES:
    print(f"{col}: {X_clean[col].nunique()}")

print("\nMissing values:")
print(X_clean.isna().sum())

print("\nAIRPORT distribution:")
print(X_clean["AIRPORT"].value_counts())

print("\nTRIPTYPE distribution:")
print(X_clean["TRIPTYPE"].value_counts())

print("\nSaved:")
print("data/processed/X_clean.csv")
print("data/processed/y_target.csv")
