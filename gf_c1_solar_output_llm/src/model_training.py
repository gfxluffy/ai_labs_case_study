"""
model_training.py

Pipeline for Challenge 1 - predictive model step.

With only ~49 cities (one row per city after aggregation), this is a
small-n regression problem, which shapes every choice below:
- A narrow, domain-informed feature set (not all ~48 mean/min/max columns)
  to avoid an almost 1:1 feature-to-sample ratio.
- A regularized linear model (Ridge/Lasso) rather than tree ensembles,
  which need more data to avoid memorizing.
- Leave-One-Out Cross-Validation (LOOCV) instead of a single train/test
  split, which would be too small/noisy to trust with ~49 rows.
- An explicit baseline ("predict the mean") comparison, so the reported
  error is meaningful rather than just impressive-looking.

Also saves the fitted model, the feature list, and each feature's observed
min/max range - the agent layer uses that range to flag when a "what-if"
query (e.g. "double the rainfall") extrapolates beyond what the model has
actually seen.

Usage:
    python model_training.py

"""

import configparser
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg = configparser.ConfigParser()
cfg.read('conf.cfg')
_mt_cfg = cfg["model_training"]

MODELING_DATA_PATH = _mt_cfg["modeling_data_path"]
CITY_COLUMN = _mt_cfg["city_column"]
TARGET_COLUMN = _mt_cfg["target_column"]

# Narrow, domain-informed feature set - variables with a direct physical
# relationship to solar output. Deliberately NOT using all mean/min/max
# columns (see module docstring: avoids near 1:1 feature-to-sample ratio).
FEATURES = [f.strip() for f in _mt_cfg["features"].split(",")]

RIDGE_ALPHA = float(_mt_cfg["ridge_alpha"])

MODEL_OUTPUT_PATH = _mt_cfg["model_output_path"]
FEATURE_RANGES_PATH = _mt_cfg["feature_ranges_path"]
CITY_FEATURES_OUTPUT_PATH = _mt_cfg["city_features_output_path"]  # used by the agent layer


# ---------------------------------------------------------------------------
# Step 1: Load and validate
# ---------------------------------------------------------------------------

def load_modeling_data(path=MODELING_DATA_PATH, features=FEATURES,
                        target=TARGET_COLUMN, city_col=CITY_COLUMN):
    df = pd.read_csv(path)

    missing_cols = [c for c in features + [target, city_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected columns: {missing_cols}. "
            f"Check modeling_data_aggregated.csv against FEATURES/TARGET_COLUMN/CITY_COLUMN."
        )

    before = len(df)
    df = df.dropna(subset=features + [target])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in the selected features/target.")

    print(f"Modeling data: {len(df)} cities, {len(features)} features")
    return df


# ---------------------------------------------------------------------------
# Step 2: LOOCV evaluation + baseline comparison
# ---------------------------------------------------------------------------

def evaluate_with_loocv(df, features=FEATURES, target=TARGET_COLUMN, alpha=RIDGE_ALPHA):
    X = df[features].values
    y = df[target].values

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    baseline_pred = np.full_like(y, y.mean(), dtype=float)
    baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))

    print("\n--- LOOCV results (Ridge) ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"\nBaseline (predict mean) RMSE: {baseline_rmse:.4f}")
    beats = rmse < baseline_rmse
    print(f"Model {'beats' if beats else 'does NOT beat'} the baseline")
    if not beats:
        print("WARNING: report this honestly in your presentation rather than hiding it - "
              "it may mean the chosen features have a weak relationship with solar output "
              "at this sample size, which is itself a valid finding to discuss.")

    return {"rmse": rmse, "mae": mae, "r2": r2, "baseline_rmse": baseline_rmse, "beats_baseline": beats}


def inspect_feature_importance(df, features=FEATURES, target=TARGET_COLUMN, cv=5, random_state=42):
    """LassoCV on the same data to see which features it keeps/zeroes -
    a useful sanity check on whether the chosen FEATURES are pulling their
    weight, even though the final saved model uses Ridge for stability."""
    X = df[features].values
    y = df[target].values
    lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, random_state=random_state, max_iter=10000))
    lasso.fit(X, y)
    coefs = lasso.named_steps["lassocv"].coef_
    print("\n--- LassoCV coefficients (feature relevance check) ---")
    for feat, coef in zip(features, coefs):
        flag = "  (zeroed out)" if coef == 0 else ""
        print(f"  {feat}: {coef:.4f}{flag}")


# ---------------------------------------------------------------------------
# Step 3: Fit final model on all data and save artifacts
# ---------------------------------------------------------------------------

def fit_final_model(df, features=FEATURES, target=TARGET_COLUMN, alpha=RIDGE_ALPHA):
    """Fit on ALL available cities (no holdout) - with only ~49 points,
    LOOCV above is your estimate of generalization error; this final model
    is what the agent actually calls at inference time."""
    X = df[features].values
    y = df[target].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X, y)
    return model


def save_artifacts(model, df, features=FEATURES, target=TARGET_COLUMN, city_col=CITY_COLUMN):
    joblib.dump({"model": model, "features": features, "target": target}, MODEL_OUTPUT_PATH)
    print(f"Saved model to {MODEL_OUTPUT_PATH}")

    # Observed min/max per feature - the agent uses this to flag
    # extrapolation on "what-if" queries (e.g. "double the rainfall").
    ranges = {
        feat: {"min": float(df[feat].min()), "max": float(df[feat].max())}
        for feat in features
    }
    with open(FEATURE_RANGES_PATH, "w") as f:
        json.dump(ranges, f, indent=2)
    print(f"Saved feature ranges to {FEATURE_RANGES_PATH}")

    # Per-city feature table - the agent looks up a city's baseline features
    # here rather than recomputing them from raw weather data each time.
    lookup_cols = [city_col] + features + [target]
    df[lookup_cols].to_csv(CITY_FEATURES_OUTPUT_PATH, index=False)
    print(f"Saved per-city feature lookup table to {CITY_FEATURES_OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Step 1: Loading modeling data...")
    df = load_modeling_data()

    print("\nStep 2: Evaluating with LOOCV...")
    evaluate_with_loocv(df)
    inspect_feature_importance(df)

    print("\nStep 3: Fitting final model on all cities and saving artifacts...")
    model = fit_final_model(df)
    save_artifacts(model, df)

    print("\nDone. Artifacts ready for the agent layer:")
    print(f"  - {MODEL_OUTPUT_PATH}")
    print(f"  - {FEATURE_RANGES_PATH}")
    print(f"  - {CITY_FEATURES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
