"""
model_comparison.py

Compares candidate regressors for Challenge 1 (Ridge, LinearRegression,
Lasso, and tree-based models) under the same LOOCV protocol used in
model_training.py, so the choice of Ridge as the production model is
backed by a side-by-side number rather than assumed.

With only ~49 cities, tree-based models are expected to underperform
(too little data to split on), but they're included here to make that
concrete rather than assumed. Interpret the table with that in mind
rather than picking the raw top-RMSE model.

Usage:
    python model_comparison.py
    
"""

import configparser

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from model_training import FEATURES, TARGET_COLUMN, load_modeling_data

cfg = configparser.ConfigParser()
cfg.read('conf.cfg')
RANDOM_STATE = int(cfg["model_comparison"]["random_state"])

MODELS = {
    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1, random_state=RANDOM_STATE)),
    "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=RANDOM_STATE),
}


def compare_models(df, models=MODELS, features=FEATURES, target=TARGET_COLUMN):
    X = df[features].values
    y = df[target].values
    loo = LeaveOneOut()

    baseline_pred = np.full_like(y, y.mean(), dtype=float)
    baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))

    results = []
    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=loo)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        results.append({
            "model": name,
            "rmse": rmse,
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "beats_baseline": rmse < baseline_rmse,
        })

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    print(f"\nBaseline (predict mean) RMSE: {baseline_rmse:.4f}\n")
    print("--- LOOCV model comparison (sorted by RMSE) ---")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return results_df


def main():
    print("Loading modeling data...")
    df = load_modeling_data()

    print("\nComparing models with LOOCV...")
    compare_models(df)


if __name__ == "__main__":
    main()
