
import pandas as pd

print("STEP 3: CATEGORICAL DATA INSPECTION")
print("=" * 60)

X = pd.read_csv("data/processed/X_features.csv")

CATEGORICAL_FEATURES = [
    "ORIGINCITY",
    "ORIGINSTATE",
    "ORIGINZIP",
    "AIRPORT",
    "TRIPTYPE"
]

for col in CATEGORICAL_FEATURES:

    print(f"\n{'=' * 50}")
    print(f"FEATURE: {col}")
    print("=" * 50)

    print("Unique values:", X[col].nunique(dropna=True))
    print("Missing values:", X[col].isna().sum())

    print("\nMost common values:")
    print(X[col].value_counts(dropna=False).head(15))

print("\nINSPECTION COMPLETED")
