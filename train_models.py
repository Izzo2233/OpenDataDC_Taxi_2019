import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from lightgbm import LGBMRegressor

print("STEP 6: TRAINING & EVALUATING 4 REGRESSION MODELS")
print("=" * 65)

# 1. Load prepared sparse matrices and target arrays
data_dir = "data/prepared"
X_train = sparse.load_npz(f"{data_dir}/X_train_processed.npz")
X_test = sparse.load_npz(f"{data_dir}/X_test_processed.npz")
y_train = np.load(f"{data_dir}/y_train.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

print(f"Loaded X_train: {X_train.shape} | X_test: {X_test.shape}\n")

# 2. Define the 4 Models (tuned for 1.77M sparse rows)
models = {
    "1. Ridge Regression": Ridge(alpha=1.0, solver="lsqr", tol=1e-4, random_state=42),
    "2. SGD Regressor": SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
    "3. Random Forest": RandomForestRegressor(
        n_estimators=60,
        max_depth=16,
        max_samples=0.20,  # Subsample 20% (~355k rows) per tree for speed & RAM safety
        n_jobs=-1,
        random_state=42
    ),
    "4. LightGBM Regressor": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        subsample_for_bin=200000,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
}

# 3. Train, Predict, and Evaluate
results = []
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MAE (min)": round(mae, 3),
        "RMSE (min)": round(rmse, 3),
        "R2 Score": round(r2, 4),
        "Train Time (s)": round(train_time, 1)
    })
    
    # Save trained model
    safe_name = name.split(". ")[1].lower().replace(" ", "_")
    joblib.dump(model, f"models/{safe_name}.joblib")
    print(f"   -> Done in {train_time:.1f}s | MAE: {mae:.3f} min | RMSE: {rmse:.3f} min | R2: {r2:.4f}\n")

# 4. Summary Leaderboard
results_df = pd.DataFrame(results).sort_values(by="RMSE (min)")
print("=" * 65)
print("MODEL COMPARISON LEADERBOARD (Sorted by RMSE)")
print("=" * 65)
print(results_df.to_string(index=False))

results_df.to_csv("models/model_comparison.csv", index=False)
print("\nSaved leaderboard to 'models/model_comparison.csv'")
