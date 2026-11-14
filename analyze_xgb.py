import joblib
import pandas as pd

print("STEP 7: XGBOOST FEATURE IMPORTANCE & INTERPRETATION")
print("=" * 65)

# 1. Load preprocessor and trained XGBoost model
preprocessor = joblib.load("data/prepared/preprocessor.joblib")
xgb_model = joblib.load("models/xgboost_regressor.joblib")

# 2. Extract and clean feature names
raw_names = preprocessor.get_feature_names_out()
clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]

# Gain importance measures average reduction in MSE per split
booster = xgb_model.get_booster()
gain_scores = booster.get_score(importance_type="gain")

# Map f0, f1... indices back to actual feature names
records = []
for idx, name in enumerate(clean_names):
    gain = gain_scores.get(f"f{idx}", 0.0)
    records.append({"Feature": name, "Gain": gain})

df_imp = pd.DataFrame(records)
total_gain = df_imp["Gain"].sum()
df_imp["Gain Share (%)"] = ((df_imp["Gain"] / total_gain) * 100).round(2)

# 3. Display Top 20 Individual Features
top_20 = df_imp.sort_values(by="Gain", ascending=False).head(20).reset_index(drop=True)
top_20.index = top_20.index + 1

print("\nTOP 20 INDIVIDUAL FEATURES (By Gain Share):")
print("-" * 65)
print(top_20[["Feature", "Gain Share (%)"]].to_string())

# 4. Aggregate back to the 11 Original Raw Features
def map_to_raw_column(feat_name):
    for prefix in ["ORIGINCITY", "ORIGINSTATE", "ORIGINZIP", "AIRPORT", "TRIPTYPE"]:
        if feat_name.startswith(prefix + "_"):
            return prefix
    return feat_name

df_imp["Original Column"] = df_imp["Feature"].apply(map_to_raw_column)
grouped_imp = (
    df_imp.groupby("Original Column", as_index=False)["Gain Share (%)"]
    .sum()
    .sort_values(by="Gain Share (%)", ascending=False)
    .reset_index(drop=True)
)
grouped_imp.index = grouped_imp.index + 1

print("\nAGGREGATED IMPORTANCE BY ORIGINAL 11 COLUMNS:")
print("-" * 65)
print(grouped_imp.to_string())

# Save reports
top_20.to_csv("models/xgb_top20_features.csv", index=False)
grouped_imp.to_csv("models/xgb_grouped_importance.csv", index=False)
print("\nSaved reports to 'models/xgb_top20_features.csv' and 'models/xgb_grouped_importance.csv'")
