import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("STEP 5: TRAIN/TEST SPLIT & PREPROCESSING PIPELINE")
print("=" * 60)

# 1. Load cleaned data
X = pd.read_csv("data/processed/X_clean.csv")
y = pd.read_csv("data/processed/y_target.csv").squeeze("columns")

# Ensure categorical columns remain strings
CATEGORICAL_FEATURES = [
    "ORIGINCITY",
    "ORIGINSTATE",
    "ORIGINZIP",
    "AIRPORT",
    "TRIPTYPE",
]

NUMERICAL_FEATURES = [
    "ORIGIN_BLOCK_LATITUDE",
    "ORIGIN_BLOCK_LONGITUDE",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
]

for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype(str)

# 2. 85% Training / 15% Testing Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print(f"Training set size : {X_train.shape[0]:,} rows (85%)")
print(f"Testing set size  : {X_test.shape[0]:,} rows (15%)")

# 3. Define Preprocessing Pipelines
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ],
    sparse_threshold=1.0,
)

# 4. Fit on Training Data ONLY, then Transform both Train and Test
print("\nFitting preprocessing pipeline on training data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 5. Save Preprocessor and Processed Arrays
output_dir = "data/prepared"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(preprocessor, f"{output_dir}/preprocessor.joblib")
sparse.save_npz(f"{output_dir}/X_train_processed.npz", X_train_processed)
sparse.save_npz(f"{output_dir}/X_test_processed.npz", X_test_processed)

np.save(f"{output_dir}/y_train.npy", y_train.to_numpy())
np.save(f"{output_dir}/y_test.npy", y_test.to_numpy())

# 6. Report Results
feature_names = preprocessor.get_feature_names_out()

print("\nPREPROCESSING COMPLETED")
print("=" * 60)
print(f"Total engineered features after One-Hot Encoding: {len(feature_names)}")
print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_test_processed shape : {X_test_processed.shape}")
print(f"Matrix format          : {type(X_train_processed).__name__}")
print(f"\nSaved artifacts in '{output_dir}/':")
print("- preprocessor.joblib")
print("- X_train_processed.npz")
print("- X_test_processed.npz")
print("- y_train.npy")
print("- y_test.npy")
