import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

print("TRAINING MODEL 4: XGBOOST REGRESSOR")
print("=" * 65)

# 1. تحميل البيانات
data_dir = "data/prepared"
X_train = sparse.load_npz(f"{data_dir}/X_train_processed.npz").astype(np.float32)
X_test = sparse.load_npz(f"{data_dir}/X_test_processed.npz").astype(np.float32)
y_train = np.load(f"{data_dir}/y_train.npy").astype(np.float32)
y_test = np.load(f"{data_dir}/y_test.npy").astype(np.float32)

# 2. إعداد نموذج XGBoost (مُحسّن للمصفوفات الكبيرة hist)
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",  # خوارزمية سريعة جداً وتستهلك ذاكرة قليلة
    n_jobs=-1,
    random_state=42
)

print("Training 4. XGBoost Regressor...")
start_time = time.time()
xgb.fit(X_train, y_train)
train_time = time.time() - start_time

y_pred = xgb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(xgb, "models/xgboost_regressor.joblib")
print(f"   -> Done in {train_time:.1f}s | MAE: {mae:.3f} min | RMSE: {rmse:.3f} min | R2: {r2:.4f}\n")

# 3. بناء جدول المقارنة النهائي للنماذج الأربعة
results = [
    {"Model": "1. Ridge Regression", "MAE (min)": 6.572, "RMSE (min)": 9.496, "R2 Score": 0.1500, "Train Time (s)": 5.7},
    {"Model": "2. SGD Regressor", "MAE (min)": 6.568, "RMSE (min)": 9.502, "R2 Score": 0.1489, "Train Time (s)": 60.5},
    {"Model": "3. Random Forest", "MAE (min)": 6.310, "RMSE (min)": 9.257, "R2 Score": 0.1923, "Train Time (s)": 633.1},
    {"Model": "4. XGBoost Regressor", "MAE (min)": round(mae, 3), "RMSE (min)": round(rmse, 3), "R2 Score": round(r2, 4), "Train Time (s)": round(train_time, 1)},
]

results_df = pd.DataFrame(results).sort_values(by="RMSE (min)")
print("=" * 65)
print("FINAL 4-MODEL COMPARISON LEADERBOARD (Sorted by RMSE)")
print("=" * 65)
print(results_df.to_string(index=False))

results_df.to_csv("models/model_comparison.csv", index=False)
print("\nSaved leaderboard to 'models/model_comparison.csv'")
