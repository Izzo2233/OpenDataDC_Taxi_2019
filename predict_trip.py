import joblib
import numpy as np
import pandas as pd

# 1. Load the trained v2 preprocessor and XGBoost model once at startup
PREPROCESSOR_PATH = "models/v2/preprocessor_v2.joblib"
MODEL_PATH = "models/v2/xgboost_v2.joblib"

preprocessor_v2 = joblib.load(PREPROCESSOR_PATH)
xgb_v2 = joblib.load(MODEL_PATH)

# 2. Helper: Haversine distance in kilometers
def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

# 3. Reusable Inference Function
def predict_trip_duration(trips_input):
    """
    Predicts DC taxi trip duration (in minutes) for one or more raw trip records.
    
    Parameters:
        trips_input (dict | list[dict] | pd.DataFrame): Raw trip feature(s) containing:
            - ORIGINCITY, ORIGINSTATE, ORIGINZIP, AIRPORT, TRIPTYPE
            - ORIGIN_BLOCK_LATITUDE, ORIGIN_BLOCK_LONGITUDE
            - hour, dayofweek, month, is_weekend
            
    Returns:
        pd.DataFrame: Original inputs alongside 'predicted_duration_min'.
    """
    if isinstance(trips_input, dict):
        df = pd.DataFrame([trips_input])
    elif isinstance(trips_input, list):
        df = pd.DataFrame(trips_input)
    else:
        df = trips_input.copy()

    # A. Clean categorical text & standardize missing placeholders
    cat_cols = ["ORIGINCITY", "ORIGINSTATE", "ORIGINZIP", "AIRPORT", "TRIPTYPE"]
    missing_labels = {"", "-", "--", "---", "----", "-----", "UNKNOWN", "NAN", "NONE", "<NA>", "0", "00000"}

    for col in cat_cols:
        df[col] = df[col].astype("string").str.strip()
        if col == "ORIGINZIP":
            df[col] = df[col].str.replace(r"\.0$", "", regex=True)
        mask = df[col].isna() | df[col].str.upper().isin(missing_labels)
        df.loc[mask, col] = "Unknown"

    # Standardize casing & Trip Type
    df["ORIGINCITY"] = df["ORIGINCITY"].apply(lambda x: x if x == "Unknown" else x.title())
    df["ORIGINSTATE"] = df["ORIGINSTATE"].apply(lambda x: x if x == "Unknown" else x.upper())
    df["TRIPTYPE"] = df["TRIPTYPE"].replace({"Transport DC": "TransportDC"})

    for col in cat_cols:
        df[col] = df[col].astype(str)

    # B. Ensure numeric types for base coordinates and calendar columns
    base_num_cols = [
        "ORIGIN_BLOCK_LATITUDE", "ORIGIN_BLOCK_LONGITUDE",
        "hour", "dayofweek", "month", "is_weekend"
    ]
    for col in base_num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # C. Recompute Step 8 spatial hub distances
    hubs = {
        "dist_dca_km": (38.8512, -77.0402),       # Reagan National Airport (DCA)
        "dist_iad_km": (38.9531, -77.4565),       # Dulles International Airport (IAD)
        "dist_union_km": (38.8977, -77.0063),     # Union Station
        "dist_downtown_km": (38.9072, -77.0369),  # Downtown DC
    }
    for col_name, (hub_lat, hub_lon) in hubs.items():
        df[col_name] = _haversine_km(
            df["ORIGIN_BLOCK_LATITUDE"],
            df["ORIGIN_BLOCK_LONGITUDE"],
            hub_lat,
            hub_lon
        )

    # D. Recompute Step 8 rush hour & cyclical time features
    df["is_rush_hour"] = (
        ((df["dayofweek"] < 5) & (df["hour"].between(7, 9) | df["hour"].between(16, 19)))
        .astype(int)
    )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # E. Transform via fitted pipeline & generate predictions
    X_proc = preprocessor_v2.transform(df).astype(np.float32)
    preds = xgb_v2.predict(X_proc)

    result = trips_input.copy() if isinstance(trips_input, pd.DataFrame) else pd.DataFrame(
        [trips_input] if isinstance(trips_input, dict) else trips_input
    )
    result["predicted_duration_min"] = np.round(preds, 2)
    return result


# 4. Example Usage / Smoke Test
if __name__ == "__main__":
    sample_trips = [
        {
            "ORIGINCITY": "Washington",
            "ORIGINSTATE": "DC",
            "ORIGINZIP": "20002",
            "AIRPORT": "N",
            "TRIPTYPE": "Ordinal",
            "ORIGIN_BLOCK_LATITUDE": 38.8977,
            "ORIGIN_BLOCK_LONGITUDE": -77.0063,
            "hour": 8,
            "dayofweek": 2,   # Wednesday morning rush hour
            "month": 5,
            "is_weekend": 0
        },
        {
            "ORIGINCITY": "Arlington",
            "ORIGINSTATE": "VA",
            "ORIGINZIP": "22202",
            "AIRPORT": "Y",
            "TRIPTYPE": "Ordinal",
            "ORIGIN_BLOCK_LATITUDE": 38.8512,
            "ORIGIN_BLOCK_LONGITUDE": -77.0402,
            "hour": 17,
            "dayofweek": 4,   # Friday evening rush hour from DCA Airport
            "month": 10,
            "is_weekend": 0
        },
        {
            "ORIGINCITY": "Washington",
            "ORIGINSTATE": "DC",
            "ORIGINZIP": "20037",
            "AIRPORT": None,    # Missing airport flag handled automatically
            "TRIPTYPE": "Transport DC",
            "ORIGIN_BLOCK_LATITUDE": 38.9005,
            "ORIGIN_BLOCK_LONGITUDE": -77.0510,
            "hour": 23,
            "dayofweek": 6,   # Sunday late night
            "month": 7,
            "is_weekend": 1
        }
    ]

    print("STEP 10: RUNNING SAMPLE INFERENCE")
    print("=" * 75)
    predictions_df = predict_trip_duration(sample_trips)
    display_cols = ["ORIGINCITY", "ORIGINSTATE", "ORIGINZIP", "AIRPORT", "hour", "predicted_duration_min"]
    print(predictions_df[display_cols].to_string(index=False))
